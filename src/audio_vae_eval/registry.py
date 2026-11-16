from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    environment: str
    sample_rate: int
    channels: int
    temporal_stride: int
    latent_channels: int
    external_vocoder: bool = False
    notes: str = ""

    @property
    def latent_frame_rate(self) -> float:
        return self.sample_rate / self.temporal_stride

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "latent_frame_rate": self.latent_frame_rate}


MODEL_REGISTRY = {
    "audioldm2": ModelSpec(
        "audioldm2", "audioldm2", 16_000, 1, 640, 8, True, "mel hop 160 x VAE stride 4"
    ),
    "mmaudio-16k": ModelSpec(
        "mmaudio-16k", "mmaudio", 16_000, 1, 512, 20, True, "optional MMAudio profile"
    ),
    "mmaudio-44k": ModelSpec(
        "mmaudio-44k", "mmaudio", 44_100, 1, 1024, 40, True, "default MMAudio profile"
    ),
    "stable-audio-open": ModelSpec(
        "stable-audio-open", "stable-audio-open", 44_100, 2, 2048, 64
    ),
    "same-s": ModelSpec("same-s", "same", 44_100, 2, 4096, 256),
    "same-l": ModelSpec("same-l", "same", 44_100, 2, 4096, 256),
    "woosh-ae": ModelSpec(
        "woosh-ae", "woosh", 48_000, 1, 480, 128, notes="centered STFT adds one frame"
    ),
}

DEFAULT_MODELS = (
    "audioldm2",
    "mmaudio-44k",
    "stable-audio-open",
    "same-s",
    "same-l",
    "woosh-ae",
)


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}; choose one of: {choices}") from error
