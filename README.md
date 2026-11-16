# Audio VAE Evaluation

AudioLDM2, MMAudio, Stable Audio Open, SAME-S/L, Woosh-AE의 복원 품질과
계산량을 같은 인터페이스로 평가하는 프레임워크입니다. 모델은 서로 충돌하는
PyTorch/Python 버전을 사용하므로 각각 독립 `uv` 환경에서 실행됩니다.

## 설치

Python 3.10/3.12와 `uv`, Git, Bazel(ViSQOL 빌드용)이 필요합니다.

```bash
cd /workspace/AudioVAE_eval
uv venv
uv pip install -e '.[dev]'
uv run audio-vae-eval bootstrap --build-visqol
```

`bootstrap`은 공식 저장소를 `third_party/`에 shallow clone하고 VAE 복원에 필요한
가중치만 `checkpoints/`에 받습니다. AudioLDM2의 8–11 GB 전체 생성 checkpoint는
받지 않습니다. Stable Audio Open은
`/workspace/TangoFlux/pretrained/declare-lab-TangoFlux/vae.safetensors`를
재사용합니다. SAME-L 가중치만 약 3.4 GB입니다.

소스/가중치/환경 단계를 선택적으로 생략할 수 있습니다.

```bash
uv run audio-vae-eval bootstrap --skip-envs
uv run audio-vae-eval bootstrap --skip-sources --skip-checkpoints
```

## 사용

```bash
audio-vae-eval list-models

audio-vae-eval reconstruct \
  --input /path/to/audio \
  --models stable-audio-open same-s \
  --output outputs

audio-vae-eval evaluate \
  --input /path/to/audio-or-manifest.jsonl \
  --output outputs

audio-vae-eval report \
  --results outputs/results \
  --output outputs/summary

# 설치된 모든 backend의 짧은 encode/decode 확인
python scripts/smoke_models.py --device cuda
```

ViSQOL 없이 나머지 metric만 먼저 확인하려면 `evaluate --no-visqol`을 사용합니다.
평가 결과는 파일별 JSON, 전체 JSONL, 집계 CSV/JSON으로 저장됩니다. 기본적으로
완료된 파일은 재실행하지 않으며 `--no-resume`으로 덮어쓸 수 있습니다.

manifest는 경로 문자열 또는 `{"path": "relative/file.wav"}` 객체의 JSON 배열/JSONL입니다.

## 지표 정의

- ViSQOL: 48 kHz mono로 변환한 뒤 공식 audio mode MOS-LQO
- SI-SDR: TorchMetrics, 기본 `zero_mean=false`
- STFT_log1p: FFT 128/256/512/1024/2048/4096, Hann, 75% overlap
- MEL_log1p: FFT 1024/2048/4096, 64 mel bands
- Band LSD / Energy Error: 0–2, 2–4, 4–8, 8–16, 16 kHz–Nyquist
- STFT-LF: 2 kHz 미만
- STFT-HF8: 8 kHz 초과(EAR-VAE2 정의)
- STFT-HF16: 16 kHz 초과의 별도 확장 지표

Nyquist보다 높은 대역은 JSON에서 `null`로 기록됩니다. AudioLDM2와 MMAudio의
외부 vocoder 파라미터는 VAE 파라미터에서 제외되며 별도 열과 vocoder ceiling
metric으로 기록됩니다. Oobleck/Woosh/SAME decoder는 waveform을 직접 합성하므로
별도 vocoder로 빼지 않습니다.

설정과 대역 경계는 `configs/eval.yaml`에서 바꿀 수 있습니다.

## 라이선스

이 프레임워크 코드는 MIT입니다. 내려받은 각 모델과 가중치에는 각 원 프로젝트의
라이선스가 적용됩니다. 특히 MMAudio/Woosh는 CC BY-NC, AudioLDM2는
CC BY-NC-SA 계열 조건을 확인해야 합니다.
