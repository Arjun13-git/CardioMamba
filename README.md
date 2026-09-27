# CardioMamba

**Mamba-Based Deep Learning for Multi-Lead ECG Classification**

CardioMamba is an educational/research-oriented deep learning project that applies a Mamba/selective state-space architecture to multi-lead ECG time-series classification using the PTB-XL dataset.

> **Medical disclaimer:** CardioMamba is a student/research project and is not a clinical diagnostic device. Model outputs must not be interpreted as medical diagnoses.

## Planned Stack
- Python + PyTorch
- Mamba / selective state-space model
- PTB-XL ECG dataset
- FastAPI inference backend
- React + Vite frontend
- NumPy / SciPy / WFDB / pandas / scikit-learn as required

## Repository Layout
```text
CardioMamba/
├── CLAUDE.md
├── System-Design/          # local architecture/design source; do not commit by default
├── ml/                     # model, preprocessing, training, evaluation
├── backend/                # FastAPI inference API
├── frontend/               # React frontend
├── scripts/                # setup and utility scripts
├── docs/                   # public documentation
└── tests/                  # integration tests
```

## Current Status
Architecture and implementation plan are defined. Dataset/model implementation is pending.

## Planned Workflow
1. Acquire PTB-XL from PhysioNet.
2. Validate metadata and ECG signal shapes.
3. Implement the documented preprocessing pipeline.
4. Implement and unit-test the Mamba model.
5. Run a tiny smoke-training experiment.
6. Run the full training configuration.
7. Evaluate using the untouched test split.
8. Package inference behind FastAPI.
9. Build the ECG visualization dashboard.
10. Run integration tests and prepare the academic demo.

## Development
See `CLAUDE.md` for agent instructions and `System-Design/` for the detailed architecture and implementation plan.
