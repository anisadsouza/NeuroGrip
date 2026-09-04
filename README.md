# Real-Time Hand Gesture Control Interface

Software starter for a college prosthetic-limb gesture-control project using windowed multi-channel sEMG signals.

The current version focuses on the software pipeline:

- signal-quality validation
- handcrafted EMG feature extraction
- calibrated gesture prediction interface
- FastAPI service for demo/integration
- unit tests for core safety behavior

Hardware control can be added later as a separate actuator layer.

## Project Structure

```text
.
├── pyproject.toml
├── src/pclm/
│   ├── api.py
│   ├── features.py
│   ├── model.py
│   ├── schemas.py
│   └── signal_quality.py
└── tests/
    ├── test_api.py
    ├── test_features.py
    └── test_signal_quality.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Run API

```bash
uvicorn pclm.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API Demo Payload

`POST /predict`

```json
{
  "subject_id": "demo-subject",
  "sampling_rate_hz": 2000,
  "channels": [
    [0.01, 0.02, 0.01, 0.0, -0.01],
    [0.02, 0.01, 0.0, -0.01, -0.02]
  ],
  "confidence_threshold": 0.6
}
```

The starter uses a deterministic placeholder model so the API works before training. Replace it with an ONNX/SVM/LSTM model in `src/pclm/model.py` once training is ready.

## Next Milestones

1. Download NinaPro DB1/DB2/DB5 after registration.
2. Add dataset loaders for the selected database files.
3. Implement LOSO splits.
4. Train SVM baseline on these features.
5. Export model to ONNX.
6. Add latency benchmark and CI gate for P95 under 10 ms.
