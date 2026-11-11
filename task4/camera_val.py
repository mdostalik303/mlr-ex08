"""Task 4 — build the camera-like validation set once, from the ordinary validation images.

The training photos are close-ups that fill the frame; the phone camera does not give you that.
This script applies to every validation image what the camera does to it — a random crop,
a rotation, and shifted colours — and saves the result as PNG under data/leaves_camera/val,
keeping the class folders. Generated with a fixed seed, so every variant of Task 4 is
measured on exactly the same pictures and the file names say what was done to each one.

    python task4/camera_val.py            # writes data/leaves_camera/val
"""
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "data" / "leaves" / "val"
DST = HERE / "data" / "leaves_camera" / "val"
SEED = 1234                       # fixed once: this set must not change between variants

# What a hand-held close-up looks like: the leaf is not perfectly centred or square in the
# frame, the picture is tilted, and the colours depend on the sky above it.
camera_like = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomRotation(25, fill=(255, 255, 255)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
])

torch.manual_seed(SEED)
source = datasets.ImageFolder(SRC)
total = 0
for path, label in source.samples:
    image = camera_like(Image.open(path).convert("RGB"))
    class_name = source.classes[label]
    out_dir = DST / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{Path(path).stem}_crop_rot_color.png"
    image.save(out_dir / name)
    total += 1

print(f"{total} images written to {DST}")
for class_name in source.classes:
    print(f"  {class_name:10s} {len(list((DST / class_name).glob('*.png')))}")
