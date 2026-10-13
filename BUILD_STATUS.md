# VisionInspect build record — 2026-10-13

Implemented the approved single-category image-defect exploration MVP in the authorized `oshada28nilendra-debug/visioninspect` repository. Initial repository inspection found only a README on main, no project instructions and no existing PRs. Work is submitted on `feat/visioninspect-mvp` for review; no merge or deployment.

## Verified locally

- Python 3.12, torch 2.9.1+cpu, torchvision 0.24.1+cpu; actual official pretrained weights downloaded successfully.
- `python -m pytest -q`: **21 passed in 14.34 seconds**, including real-model inference and invalid-image, duplicate/leakage, invalid-label, mathematical-distance, no-evaluation-in-bank and model-download-failure cases.
- `python visioninspect.py demo --output reports/synthetic-evaluation.json`: completed with 12 normal reference images and 24 held-out synthetic washer images (12 normal/12 defect). AUROC **1.0**, average precision **1.0**, measured analysis time **1.6387 seconds**, excluding model initialization/download.
- Chromium browser test passed: real-model sample, multipart upload, JSON export, malformed-label feedback and 390-pixel mobile layout; zero JavaScript page errors. Desktop/mobile screenshots inspected locally; CI retains screenshots as artifacts.
- Sample generator and CLI folder import exercised separately; compare scores with the committed demo report.

The benchmark is a small, deliberately easy synthetic integration fixture. It does not estimate real factory performance. Overlay localization has not been measured against segmentation masks. No user adoption, production accuracy, deployment or live hardware claims are made.

## Review and CI

Workflow `.github/workflows/test.yml` repeats Python tests, the pretrained-model demo and Chromium browser tests on Python 3.11. Remote CI results must be read on the PR checks; this record does not claim a remote result before execution. The measured JSON report contains per-image scores and pixel hashes. All code and status records belong in this repository.
