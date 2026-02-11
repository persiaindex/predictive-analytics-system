# Predictive Analytics System (CPU-friendly)

End-to-end machine learning project designed for real-world workflow:
data validation → training pipeline → evaluation → inference API → Docker → CI.

## Goals
- Build a reproducible ML pipeline in Python
- Track experiments and results clearly
- Provide a FastAPI inference service (CPU)
- Use tests + linting + GitHub Actions CI

## Repo Structure
- src/ - Python package source code
- notebooks/ - EDA and experiments
- configs/ - configuration files (YAML)
- scripts/ - helper scripts / CLI entrypoints
- tests/ - automated tests (pytest)
- docs/ - documentation and presentation notes
- data/ - local-only data (not committed)
- reports/ - experiment reports and summaries
