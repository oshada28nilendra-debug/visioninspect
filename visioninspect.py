"""Local, single-category normal-patch anomaly exploration."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from sklearn.metrics import average_precision_score, roc_auc_score

MAX_BYTES = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 10_000_000


def decode(data):
    if len(data) > MAX_BYTES:
        raise ValueError('Each image must be at most 5 MiB.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ('PNG', 'JPEG') or im.width * im.height > 10_000_000:
                raise ValueError('Use PNG/JPEG images of at most 10 million pixels.')
            if min(im.size) < 32:
                raise ValueError('Images must be at least 32 pixels on each side.')
            return im.convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Cannot decode image.') from exc


def fingerprint(im):
    return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


class Encoder:
    def __init__(self):
        import torch
        from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
        self.torch = torch
        torch.set_num_threads(2)
        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1).features[:6].eval()

    def __call__(self, im):
        t = self.torch
        x = np.asarray(im.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        x = (x - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        with t.inference_mode():
            y = self.model(t.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0))[0]
        return y.permute(1, 2, 0).numpy()


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Features must be a nonempty finite matrix.')
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def distances(features, bank):
    """Nearest normalized Euclidean patch distance; bounded temporary matrix."""
    x, b = normalize(features), normalize(bank)
    if x.shape[1] != b.shape[1]:
        raise ValueError('Feature dimensions differ.')
    return np.concatenate([np.sqrt(np.maximum(0, (v*v).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*v@b.T)).min(1) for v in np.array_split(x, max(1, (len(x)+255)//256))])


def png(im):
    out = io.BytesIO()
    im.save(out, format='PNG')
    return out.getvalue()


def preview(im, grid=None):
    im = im.resize((224, 224))
    if grid is not None:
        # Fixed 0..1 display scale across images, saturating distances above 1.
        heat = np.asarray(Image.fromarray(grid.astype(np.float32)).resize(im.size, Image.Resampling.BILINEAR))
        alpha = np.clip(heat, 0, 1)[..., None] * .8
        im = Image.fromarray((np.asarray(im)*(1-alpha) + np.array([255, 30, 10])*alpha).astype('uint8'))
    return 'data:image/png;base64,' + base64.b64encode(png(im)).decode()


def analyze(reference, evaluation, encoder, labels=None, include_images=False):
    if not 2 <= len(reference) <= 30 or not 1 <= len(evaluation) <= 50:
        raise ValueError('Use 2–30 normal reference images and 1–50 evaluation images.')
    if len({n for n, _ in evaluation}) != len(evaluation):
        raise ValueError('Evaluation filenames must be unique.')
    if labels is not None and (not isinstance(labels, dict) or set(labels) != {n for n, _ in evaluation} or any(type(v) is not int or v not in (0, 1) for v in labels.values())):
        raise ValueError('Labels must map every evaluation filename to integer 0 (normal) or 1 (defect).')
    started = time.perf_counter()
    refs = [(n, decode(data)) for n, data in reference]
    tests = [(n, decode(data)) for n, data in evaluation]
    hashes = [fingerprint(im) for _, im in refs]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Reference images contain exact decoded duplicates.')
    test_hashes = [fingerprint(im) for _, im in tests]
    if set(hashes) & set(test_hashes):
        raise ValueError('Reference and evaluation images overlap; separate these sets.')
    if len(set(test_hashes)) != len(test_hashes):
        raise ValueError('Evaluation images contain exact decoded duplicates.')
    features = [encoder(im) for _, im in refs]
    bank = np.concatenate([f.reshape(-1, f.shape[-1]) for f in features])
    # Deterministic uniform subsampling; no evaluation images influence bank.
    bank = bank[np.linspace(0, len(bank)-1, min(2048, len(bank)), dtype=int)]
    rows = []
    for (name, im), digest in zip(tests, test_hashes):
        f = encoder(im)
        grid = distances(f.reshape(-1, f.shape[-1]), bank).reshape(f.shape[:2])
        row = {'name': name, 'sha256_pixels': digest, 'score': float(np.quantile(grid, .99)), 'patch_shape': list(grid.shape)}
        if labels is not None:
            row['label'] = labels[name]
        if include_images:
            row.update(original=preview(im), overlay=preview(im, grid))
        rows.append(row)
    metrics = None
    if labels is not None and len(set(labels.values())) == 2:
        y, s = [r['label'] for r in rows], [r['score'] for r in rows]
        metrics = {'auroc': float(roc_auc_score(y, s)), 'average_precision': float(average_precision_score(y, s))}
    return {'model': 'SqueezeNet1_1 IMAGENET1K_V1, features[:6]', 'score_definition': '99th percentile nearest L2-normalized normal-patch Euclidean distance', 'reference_count': len(refs), 'reference_hashes': hashes, 'bank_patches': len(bank), 'metrics': metrics, 'elapsed_seconds_excluding_model_load': time.perf_counter()-started, 'results': sorted(rows, key=lambda r: (-r['score'], r['name']))}


def sample():
    """Original CC0 synthetic washers; disjoint RNG streams, not industrial data."""
    def make(seed, defect):
        rng = np.random.default_rng(seed)
        bg = rng.normal(45, 2, (224, 224, 3)).clip(0, 255).astype('uint8')
        im = Image.fromarray(bg)
        d = ImageDraw.Draw(im)
        x, y = rng.integers(-4, 5, 2)
        c = int(rng.integers(165, 190))
        d.ellipse((35+x, 35+y, 189+x, 189+y), fill=(c,c,c), outline=(210,210,210), width=3)
        d.ellipse((82+x, 82+y, 142+x, 142+y), fill=(45,45,45), outline=(120,120,120), width=2)
        if defect:
            angle = rng.uniform(0, 2*np.pi)
            cx, cy = 112+int(55*np.cos(angle)), 112+int(55*np.sin(angle))
            d.line((cx-15,cy-12,cx+15,cy+12), fill=(35,35,35), width=7)
        return png(im)
    refs = [(f'reference_{i:02}.png', make(100+i, False)) for i in range(12)]
    tests = [(f'{"defect" if i>=12 else "normal"}_{i:02}.png', make(1000+i, i>=12)) for i in range(24)]
    return refs, tests, {n: int(i>=12) for i, (n, _) in enumerate(tests)}


def create_app(encoder=None):
    from flask import Flask, request, jsonify, render_template
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024
    holder = [encoder]

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.post('/analyze')
    def run():
        try:
            if request.form.get('sample') == 'true':
                refs, tests, labels = sample()
            else:
                refs = [(f.filename, f.read(MAX_BYTES+1)) for f in request.files.getlist('reference')]
                tests = [(f.filename, f.read(MAX_BYTES+1)) for f in request.files.getlist('evaluation')]
                raw = request.form.get('labels', '').strip()
                labels = json.loads(raw) if raw else None
            if holder[0] is None:
                try:
                    holder[0] = Encoder()
                except (OSError, RuntimeError) as exc:
                    app.logger.error('Pretrained model initialization failed: %s', exc)
                    return jsonify(error='Pretrained model unavailable. Check network access to download.pytorch.org and the local PyTorch cache, then retry.'), 503
            result = analyze(refs, tests, holder[0], labels, include_images=True)
            result['synthetic_sample'] = request.form.get('sample') == 'true'
            return jsonify(result)
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error='Request exceeds 40 MiB.'), 413

    @app.after_request
    def headers(response):
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['demo', 'analyze', 'serve', 'sample'])
    p.add_argument('--reference', type=Path)
    p.add_argument('--evaluation', type=Path)
    p.add_argument('--labels', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/report.json'))
    a = p.parse_args()
    if a.command == 'serve':
        create_app().run(host='127.0.0.1', port=8769, debug=False)
        return
    try:
        if a.command in ('sample', 'demo'):
            refs, tests, labels = sample()
        else:
            if not a.reference or not a.evaluation:
                p.error('--reference and --evaluation are required')
            def read(folder):
                paths = sorted(x for x in folder.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png'))
                items = []
                for x in paths:
                    with x.open('rb') as stream:
                        items.append((x.name, stream.read(MAX_BYTES+1)))
                return items
            refs, tests = read(a.reference), read(a.evaluation)
            labels = json.loads(a.labels.read_text()) if a.labels else None
        if a.command == 'sample':
            for split, items in [('reference', refs), ('evaluation', tests)]:
                (a.output/split).mkdir(parents=True, exist_ok=True)
                for name, data in items:
                    (a.output/split/name).write_bytes(data)
            (a.output/'labels.json').write_text(json.dumps(labels, indent=2)+'\n')
            return
        result = analyze(refs, tests, Encoder(), labels)
        result['synthetic_sample'] = a.command == 'demo'
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('results','reference_hashes')}, indent=2))
    except (ValueError, OSError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
