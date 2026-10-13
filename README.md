# VisionInspect

A local computer-vision exploration tool for developers or quality analysts with normal images of **one category**. Rank held-out images by distance from pretrained normal-patch features, review coarse suspicious regions, and export JSON evidence. This does not decide whether a manufactured part is safe or acceptable.

## Run locally

Linux, Python 3.11 or 3.12; CPU only. No accounts, GPU, or paid API required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python visioninspect.py serve
# Open http://127.0.0.1:8769 and choose “Try synthetic washer sample”.
```

The first analysis downloads official pretrained SqueezeNet1.1 weights (~4.7 MB); PyTorch dependencies are substantially larger. Weights are cached under the normal PyTorch cache (`TORCH_HOME` can relocate it). Inference then works offline. A blocked weight download is a failure, never a silent fallback to random weights. Keep the default loopback binding: this development server is not an authenticated hosted service.

CLI and tests:

```bash
python visioninspect.py demo --output outputs/report.json
python visioninspect.py sample --output outputs/sample
python visioninspect.py analyze --reference outputs/sample/reference \
  --evaluation outputs/sample/evaluation --labels outputs/sample/labels.json \
  --output outputs/imported-report.json
python -m pytest -q
```

The sample command materializes input PNGs and labels. For your own data, provide **only normal images** in reference and separate held-out normal/defect images in evaluation. CLI reads PNG/JPEG files directly in each folder; it does not recurse. Optional labels JSON maps every evaluation basename to integer `0` (normal) or `1` (defect), e.g. `{"normal.png": 0, "scratch.png": 1}`. Labels never influence fitting or scores. Without labels the tool still ranks images; metrics require both classes. UI labels are pasted into the labelled text area.

Limits: 2–30 references, 1–50 evaluation images, 5 MiB and 10 million pixels per image, minimum 32×32; web requests 40 MiB total. Duplicate decoded RGB images within either split and exact decoded reference/evaluation overlaps are rejected. This is not a near-duplicate detector: resizes/crops and related captures must be separated by the operator. Avoid unrelated categories in the same analysis. Web uploads are not persisted.

## Architecture and score

1. Pillow decodes PNG/JPEG and converts to RGB. Resize the whole image to 224×224 (aspect ratio may stretch), then ImageNet-normalize. Avoiding the classification transform's center crop preserves edge content.
2. Frozen **SqueezeNet1_1 IMAGENET1K_V1**, `features[:6]`, produces 27×27×128 spatial features. No training or fine-tuning occurs.
3. Normal reference patches form a bank, deterministically subsampled to at most 2,048 patches and L2-normalized.
4. Each evaluation patch receives its nearest reference patch's Euclidean distance. Chunks of at most 256 patches bound working memory. A zero feature vector stays zero.
5. Image score is the **99th percentile** of patch distances. Higher means less similar to the normal bank; it is neither probability nor a binary decision.
6. Upsampled patch distances make an orange overlay: opacity is `0.8 × clip(distance, 0, 1)` on the same display scale for every image. Distances above one saturate. Regions have receptive-field blur and are **not segmentation masks**.

Pooling patch positions can miss unusual arrangements of familiar patches. Early visual features react to illumination, framing and background as well as defects. The bank cap trades fidelity for CPU/memory cost. Flask serves plain JavaScript/CSS; analysis lives in `visioninspect.py`. Reports include model identity, score definition, reference/evaluation pixel hashes, timings and ranking metrics. Browser JSON exports omit large preview data URLs.

## Evaluation and data provenance

`sample()` creates **original synthetic washer fixtures**, dedicated to CC0-1.0. No external image dataset, factory data or MVTec data is bundled. Twelve normal references use seeds 100–111. Held-out evaluation uses independent seeds 1000–1023: 12 normal washers and 12 with a dark line simulating a scratch. No shared base image crosses splits. This simple category is an integration demonstration, not a representative inspection benchmark.

The architecture, 2,048-patch cap and 99th-percentile aggregation are fixed design choices. No threshold is tuned on held-out labels. AUROC measures ranking of normal versus defective images; average precision summarizes precision/recall ranking with defect as positive. There is no pixel-level evaluation or calibrated operating threshold. These 24 samples cannot establish industrial accuracy or generalization. A credible next evaluation needs independently acquired licensed real images, capture-group splits, more normal variation, multiple defect types, and separate validation for parameter/threshold choices.

See [measured report](reports/synthetic-evaluation.json) and [build status](BUILD_STATUS.md). Timings exclude model initialization/download and vary by machine; they include decoding, extraction and scoring. CI repeats real pretrained inference and saves its own report artifact. Unit tests use a fake encoder only for plumbing/math; `tests/test_model.py` runs the actual pretrained network.

## Interview explanation

“I built a normal-only visual reference model around a frozen pretrained CNN. Patch distances make image rankings inspectable, and a bounded reference bank makes CPU inference practical. I separated reference and evaluation images, rejected exact split leakage, tested failure paths, and reported a reproducible synthetic benchmark with explicit limits. Pretrained appearance differences are not the same thing as real defects.”

## Sources and licenses

- [Official model implementation and weights metadata](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/squeezenet.py)
- [Official model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_1.html)
- [torchvision BSD-3-Clause license](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE)
- [ImageNet terms](https://www.image-net.org/download.php): weights derive from ImageNet; upstream training images are not redistributed. No blanket commercial clearance for upstream data is claimed.
- Code: MIT (LICENSE). Original generated fixtures: [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/). Upstream model/dependency licenses remain their authors'.
