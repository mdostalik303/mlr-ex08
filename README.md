# mlr-ex08 — a leaf classifier in your phone's browser

**The page: <https://mdostalik303.github.io/mlr-ex08/>**
(also as a QR code at the bottom of the page, for opening it on the phone itself)

Point the phone's rear camera at a leaf and the page classifies it live into
`apple`, `cherry`, `chestnut` or `maple`. The network is my exported
`leaf_model.onnx` from Task 2, running in the browser tab with
[onnxruntime-web](https://onnxruntime.ai/) (WASM). Nothing is uploaded: the only
requests the page makes are for itself, the runtime from jsDelivr, the model, its
metadata and the two `check_*` files.

## What is in here

| file | what it is |
| --- | --- |
| `index.html` | the whole app: live camera, upload fallback, self-test, QR code |
| `leaf_model.onnx` | my trained classifier, input `image` → output `logits`, `1×3×96×96` float32 |
| `model_info.json` | `classes`, `input_size`, `mean`, `std` — read by the page at startup, nothing is hardcoded |
| `check_image.jpg`, `check_expected.json` | the notebook's reference image and its probabilities |
| `make_qr.py` | build-time script that draws the QR code into `index.html` and reads it back |

## How the frame becomes a tensor

All of it is in `index.html`, function `tensorFrom()`:

1. the biggest **central square** of the frame: `side = min(w, h)`, offset
   `((w − side)/2, (h − side)/2)`;
2. that square is drawn straight into a `96 × 96` canvas (`input_size` comes from
   `model_info.json`), so crop and resize happen in one `drawImage` with high-quality
   smoothing;
3. `getImageData` returns row-major RGBA, each byte divided by **255** → 0…1;
4. every channel is standardized with the metadata, `(x − mean[c]) / std[c]`
   with `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`;
5. the values are written into three consecutive planes — **CHW**, `1×3×96×96`, plane `c`
   starting at `c·96·96` — which is the notebook's `ToTensor()` + `Normalize()` on a centre crop.

The four `logits` are turned into probabilities with a softmax. The same `tensorFrom()`
serves the camera and an uploaded photo, so the two cannot drift apart.

## Checking it against the notebook

Open the page: section **3. Check against the notebook** runs by itself. It decodes
`check_image.jpg`, classifies it and prints its numbers next to `check_expected.json`.
Measured in Chrome 157 (Windows): the largest difference is **1.2·10⁻⁶** — the only thing
that differs from the notebook is the JPEG decoder. Running `check_image.jpg` through the
upload button gives apple **98.34 %**, cherry **1.66 %**, chestnut 0.00 %, maple 0.00 %.

Two more checks were run against this page in headless Chrome, with synthetic camera
streams handed to `getUserMedia`:

- **the crop really is the central square** — on a 640×480 frame that is pure red except
  for a pure green central 480×480 square, `tensorFrom()` returned `1×3×96×96` in which
  every pixel reads back (after undoing the normalization) as exactly `R 0, G 255, B 0`:
  no red leaked in, so the crop is centred, and the `/255` + mean/std step is exact;
- **the live path equals the file path** — on a 96×96 stream showing `check_image.jpg`
  pixel for pixel, the live loop printed apple 98.34 % / cherry 1.66 %, the same numbers
  as the upload, with the frames bit-identical to the source image.

## Notes and limits

- Needs **HTTPS** (or localhost) — `getUserMedia` refuses on plain http; GitHub Pages is https.
- iPhone: **iOS 16.4 or newer**, because the onnxruntime-web WASM build uses SIMD, which
  Safari gained then. The page detects the missing feature and says so instead of hanging.
- `ort.env.wasm.numThreads = 1` — GitHub Pages sends no COOP/COEP headers, so
  `SharedArrayBuffer` is unavailable and a threaded runtime would never start.
- The camera starts on a tap (browsers require a gesture), the preview is a square with
  `object-fit: cover` so what you see is exactly the region that gets classified, and the
  stream is stopped when the tab is hidden so the camera light goes out.
- **The QR code** at the bottom is a vector graphic inlined in `index.html`, drawn offline by
  `make_qr.py` — `qrcode` builds the matrix, `zxing-cpp` reads the result back and confirms it
  says `https://mdostalik303.github.io/mlr-ex08/`. No QR web service is contacted, at build
  time or in the browser. `python -I make_qr.py --check` re-verifies the code that ships.
- Resampling really does move this model: the same pipeline applied in Python to
  `check_image.jpg` first upscaled 5× to 480×480 lands at apple 45.5 % / cherry 54.5 %
  instead of 98.3 % / 1.7 %. That is the network's sensitivity to a blurry, enlarged leaf,
  not a difference between the two code paths — a sharp close-up is what it was trained on.

## Serving it yourself

GitHub Pages publishes the `main` branch as it is. Locally:

```
python -m http.server 8000     # then open http://localhost:8000/
```
