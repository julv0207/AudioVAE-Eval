from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio

from audio_vae_eval.audio import load_audio, pad_to_multiple, save_audio
from audio_vae_eval.registry import get_model_spec


def parameter_count(module: torch.nn.Module | None) -> int:
    return 0 if module is None else sum(parameter.numel() for parameter in module.parameters())


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu")
    return torch.device(requested)


def vocoder_waveform(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    waveform = getattr(output, "waveform", None)
    if isinstance(waveform, torch.Tensor):
        return waveform
    raise TypeError(f"Unsupported vocoder output type: {type(output).__name__}")


def audioldm2(
    waveform: torch.Tensor, root: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    from diffusers import AutoencoderKL
    from transformers import SpeechT5HifiGan

    checkpoint = root / "checkpoints" / "audioldm2"
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae", local_files_only=True)
    vocoder = SpeechT5HifiGan.from_pretrained(
        checkpoint, subfolder="vocoder", local_files_only=True
    )
    vae = vae.eval().to(device)
    vocoder = vocoder.eval().to(device)

    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    peak = waveform.abs().amax()
    if peak > 0:
        waveform = waveform * (0.5 / peak)
    waveform, original_length = pad_to_multiple(waveform, 640)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        f_min=0,
        f_max=8_000,
        n_mels=64,
        power=1.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="slaney",
    ).to(device)
    audio = waveform.to(device)
    mel = torch.log(torch.clamp(transform(audio), min=1e-5)).transpose(-1, -2)
    target_frames = math.ceil(waveform.shape[-1] / 160)
    mel = mel[:, :target_frames]
    mel = torch.nn.functional.pad(mel, (0, 0, 0, target_frames - mel.shape[1]))
    vae_input = mel.unsqueeze(0)
    with torch.inference_mode():
        latent = vae.encode(vae_input).latent_dist.mode()
        decoded_mel = vae.decode(latent).sample[:, 0]
        reconstructed = vocoder_waveform(vocoder(decoded_mel))
        ceiling = vocoder_waveform(vocoder(mel))
    reconstructed = reconstructed.reshape(1, -1)[:, :original_length]
    ceiling = ceiling.reshape(1, -1)[:, :original_length]
    return reconstructed, latent, {
        "prepared_reference": waveform[:, :original_length].detach().cpu(),
        "vae_parameters": parameter_count(vae),
        "encoder_parameters": parameter_count(vae.encoder),
        "decoder_parameters": parameter_count(vae.decoder),
        "vocoder_parameters": parameter_count(vocoder),
        "vocoder_ceiling": ceiling.detach().cpu(),
    }


def _call_mel_converter(converter: object, waveform: torch.Tensor) -> torch.Tensor:
    for name in ("waveform_to_mel", "get_mel", "mel_spectrogram"):
        function = getattr(converter, name, None)
        if callable(function):
            return function(waveform)
    if callable(converter):
        return converter(waveform)
    raise RuntimeError("Unsupported MMAudio MelConverter API")


def mmaudio(
    waveform: torch.Tensor, root: Path, device: torch.device, mode: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = root / "third_party" / "mmaudio"
    sys.path.insert(0, str(source))
    from mmaudio.ext.autoencoder.vae import get_my_vae
    from mmaudio.ext.bigvgan import BigVGAN
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
    from mmaudio.ext.mel_converter import get_mel_converter

    weights = root / "checkpoints" / "mmaudio" / "ext_weights"
    vae_path = weights / ("v1-16.pth" if mode == "16k" else "v1-44.pth")
    vae = get_my_vae(mode).eval()
    vae.load_state_dict(torch.load(vae_path, weights_only=True, map_location="cpu"))
    vae.remove_weight_norm()
    if mode == "16k":
        vocoder = BigVGAN(str(weights / "best_netG.pt")).eval()
    else:
        vocoder = BigVGANv2.from_pretrained(
            str(root / "checkpoints" / "mmaudio-bigvgan"),
            use_cuda_kernel=False,
        )
        vocoder.remove_weight_norm()
    vae = vae.to(device)
    vocoder = vocoder.eval().to(device)
    converter = get_mel_converter(mode).eval().to(device)
    stride = 512 if mode == "16k" else 1024
    padded, original_length = pad_to_multiple(waveform, stride)
    audio = padded.to(device)
    with torch.inference_mode():
        mel = _call_mel_converter(converter, audio)
        posterior = vae.encode(mel)
        latent = posterior.mode()
        decoded_mel = vae.decode(latent)
        reconstructed = vocoder(decoded_mel)
        ceiling = vocoder(mel)
    reconstructed = reconstructed.reshape(1, -1)[:, :original_length]
    return reconstructed, latent, {
        "prepared_reference": waveform.detach().cpu(),
        "vae_parameters": parameter_count(vae),
        "encoder_parameters": parameter_count(vae.encoder),
        "decoder_parameters": parameter_count(vae.decoder),
        "vocoder_parameters": parameter_count(vocoder),
        "vocoder_ceiling": ceiling.reshape(1, -1)[:, :original_length].detach().cpu(),
    }


def stable_audio_open(
    waveform: torch.Tensor, root: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    from diffusers import AutoencoderOobleck
    from safetensors.torch import load_file

    model = AutoencoderOobleck()
    checkpoint = root.parent / "TangoFlux" / "pretrained" / "declare-lab-TangoFlux"
    model.load_state_dict(load_file(str(checkpoint / "vae.safetensors")))
    model = model.eval().to(device)
    padded, original_length = pad_to_multiple(waveform, 2048)
    with torch.inference_mode():
        latent = model.encode(padded.unsqueeze(0).to(device)).latent_dist.mode()
        reconstructed = model.decode(latent).sample[0, :, :original_length]
    encoder_parameters = parameter_count(model.encoder)
    decoder_parameters = parameter_count(model.decoder)
    return reconstructed, latent, {
        "prepared_reference": waveform.detach().cpu(),
        "encoder_parameters": encoder_parameters,
        "decoder_parameters": decoder_parameters,
        "vae_parameters": parameter_count(model),
        "vocoder_parameters": 0,
    }


def same(
    waveform: torch.Tensor, root: Path, device: torch.device, size: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = root / "third_party" / "stable-audio-3"
    sys.path.insert(0, str(source))
    import stable_audio_3.model as same_model
    from stable_audio_3 import AutoencoderModel

    model_name = f"same-{size}"
    checkpoint = root / "checkpoints" / model_name

    class LocalAutoencoderConfig:
        def resolve(self) -> tuple[str, str]:
            return (
                str(checkpoint / "model_config.json"),
                str(checkpoint / "model.safetensors"),
            )

    same_model.ae_models[model_name] = LocalAutoencoderConfig()
    model = AutoencoderModel.from_pretrained(model_name, device=str(device))
    padded, original_length = pad_to_multiple(waveform, 4096)
    with torch.inference_mode():
        latent = model.encode(padded.to(device), 44_100)
        reconstructed = model.decode(latent)[0, :, :original_length]
    module = model.autoencoder
    encoder = getattr(module, "encoder", None)
    decoder = getattr(module, "decoder", None)
    return reconstructed, latent, {
        "prepared_reference": waveform.detach().cpu(),
        "encoder_parameters": parameter_count(encoder),
        "decoder_parameters": parameter_count(decoder),
        "vae_parameters": parameter_count(module),
        "vocoder_parameters": 0,
    }


def woosh(
    waveform: torch.Tensor, root: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = root / "third_party" / "woosh"
    sys.path.insert(0, str(source))
    from woosh.components.autoencoders import AudioAutoEncoder
    from woosh.components.base import LoadConfig

    model = AudioAutoEncoder(LoadConfig(path=str(root / "checkpoints" / "woosh-ae")))
    model.load_from_config()
    model = model.eval().to(device)
    with torch.inference_mode():
        latent = model(waveform.unsqueeze(0).to(device))
        reconstructed = model.inverse(latent)[0, :, : waveform.shape[-1]]
    module = getattr(model, "autoencoder", model)
    return reconstructed, latent, {
        "prepared_reference": waveform.detach().cpu(),
        "encoder_parameters": parameter_count(getattr(module, "encoder", None)),
        "decoder_parameters": parameter_count(getattr(module, "decoder", None)),
        "vae_parameters": parameter_count(module),
        "vocoder_parameters": 0,
    }


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    name = request["model"]
    spec = get_model_spec(name)
    root = Path(request["root"])
    device = choose_device(request["device"])
    waveform, _ = load_audio(
        request["input_path"], sample_rate=spec.sample_rate, channels=spec.channels
    )
    if name == "audioldm2":
        reconstruction, latent, details = audioldm2(waveform, root, device)
    elif name.startswith("mmaudio-"):
        reconstruction, latent, details = mmaudio(waveform, root, device, name.split("-")[1])
    elif name == "stable-audio-open":
        reconstruction, latent, details = stable_audio_open(waveform, root, device)
    elif name.startswith("same-"):
        reconstruction, latent, details = same(waveform, root, device, name[-1])
    elif name == "woosh-ae":
        reconstruction, latent, details = woosh(waveform, root, device)
    else:
        raise ValueError(name)

    save_audio(request["output_path"], reconstruction, spec.sample_rate)
    prepared_reference = details.pop("prepared_reference", waveform)
    prepared_reference_path = str(
        Path(request["output_path"]).with_suffix(".prepared_reference.wav")
    )
    save_audio(prepared_reference_path, prepared_reference, spec.sample_rate)
    ceiling = details.pop("vocoder_ceiling", None)
    ceiling_path = None
    if ceiling is not None:
        ceiling_path = str(Path(request["output_path"]).with_suffix(".vocoder_ceiling.wav"))
        save_audio(ceiling_path, ceiling, spec.sample_rate)
    metadata = {
        **spec.to_dict(),
        **details,
        "input_samples": waveform.shape[-1],
        "output_samples": reconstruction.shape[-1],
        "latent_shape": list(latent.shape),
        "checkpoint_policy": "official component-only checkpoint",
        "prepared_reference_path": prepared_reference_path,
        "vocoder_ceiling_path": ceiling_path,
    }
    Path(request["metadata_path"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_request(request)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
