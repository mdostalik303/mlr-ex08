"""Task 4 — export the chosen model for the phone, exactly as 2e exported the Task 2 one.

    python task4/export.py --variant "cnn best" --seed 0

Writes leaf_model_task4.onnx with the model_info metadata inside, and prints how far the
exported network is from the PyTorch one on the check image and on the whole validation set.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from PIL import Image
import onnxruntime
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# python -I does not put this script's own folder on sys.path, so "import train" needs it added.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import HERE, LeafNet, VAL, VARIANTS, mobilenet, plain   # the same pipeline, no copies

# torch's exporter reports success with a check mark, which the Windows console code page
# (cp1250 here) cannot encode — the export then crashes after having worked. UTF-8 output fixes it.
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="cnn best")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cfg = VARIANTS[args.variant]
weights = Path(__file__).resolve().parent / "weights" / f"{args.variant.replace(' ', '_')}_seed{args.seed}.pt"
model = (mobilenet(cfg["freeze_backbone"]) if cfg["pretrained"] else LeafNet(**cfg["net"]))
model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
model.eval()
print(f"{args.variant} (seed {args.seed}) from {weights.name}: "
      f"{sum(p.numel() for p in model.parameters()):,} parameters")

classes = VAL.dataset.classes                       # the order the labels were trained in
model_info = {"classes": classes, "input_size": 96,
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
out = HERE / "leaf_model_task4.onnx"
model.eval().cpu()
torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), out,
                  input_names=["image"], output_names=["logits"], external_data=False)
onnx_model = onnx.load(str(out))
onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
onnx.save(onnx_model, str(out))
print(f"{out.name}: {out.stat().st_size / 1e6:.1f} MB, classes {classes}")

# The exported file must be the network that was trained, not a near miss.
session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


check = plain(Image.open(HERE / "ex08_for_agent" / "check_image.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    torch_prob = softmax(model(check).numpy()[0])
onnx_prob = softmax(session.run(None, {"image": check.numpy()})[0][0])
print("\ncheck_image.jpg   torch:", "  ".join(f"{c} {p:.4f}" for c, p in zip(classes, torch_prob)))
print("check_image.jpg    onnx:", "  ".join(f"{c} {p:.4f}" for c, p in zip(classes, onnx_prob)))

# One image at a time: like the 2e export, the exported graph takes exactly 1x3x96x96, which is
# also what the browser feeds it.
loader = DataLoader(datasets.ImageFolder(HERE / "data" / "leaves" / "val", transform=plain), batch_size=1)
worst, agree, total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in loader:
        t = model(images).numpy()
        o = session.run(None, {"image": images.numpy()})[0]
        worst = max(worst, float(np.abs(t - o).max()))
        agree += int((t.argmax(1) == o.argmax(1)).sum())
        total += len(labels)
print(f"\nvalidation set: biggest logit difference {worst:.2e}, same class on {agree}/{total} images")
