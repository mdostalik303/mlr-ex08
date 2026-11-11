"""Task 4 — what actually helps the leaf classifier? One change at a time.

Starts from the Task 3 pipeline (LeafNet on data/leaves, Adam 1e-3, 15 epochs) and changes
exactly one thing per variant. Every variant sees the same training images in the same order
(the loader's shuffle is seeded) and the run is seeded, so a difference is the change and not
the luck. Every variant is scored three ways:

  val       data/leaves/val          — the course validation set, the same photos as training
  camera    data/leaves_camera/val   — the same images cropped, rotated and re-coloured once
                                       with a fixed seed by task4/camera_val.py
  others    data_history/leaves/val  — last year's students, other phones (never trained on,
                                       unless the variant is explicitly "+historical data")

    python task4/train.py                      # every variant, seeds 0 1 2
    python task4/train.py --only baseline,"phone best"   # a few of them
    python task4/train.py --seeds 0            # one seed while exploring

Writes task4/results_task4_runs.csv (one line per run, appended, so an interrupted sweep can
be continued) and rewrites task4/results_task4.csv and .md with the table for REPORT.md.
"""
import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms

HERE = Path(__file__).resolve().parent.parent          # the folder with data/ and data_history/
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH = 32
TASK4 = Path(__file__).resolve().parent
RUNS = TASK4 / "results_task4_runs.csv"                       # one line per run, appended
SUMMARY_CSV, SUMMARY_JSON, SUMMARY_MD = (TASK4 / "results_task4.csv", TASK4 / "results_task4.json",
                                         TASK4 / "results_task4.md")
RUN_COLUMNS = ["variant", "seed", "epochs", "parameters", "val", "camera", "others",
               "best_val", "train_loss", "seconds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------- the network
class LeafNet(nn.Module):
    """The Task 3 network. The three arguments default to exactly what Task 3 trained."""

    def __init__(self, bn=False, gap=False, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) if bn else None
        self.bn2 = nn.BatchNorm2d(32) if bn else None
        self.bn3 = nn.BatchNorm2d(64) if bn else None
        self.bn4 = nn.BatchNorm2d(64) if bn else None
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = gap                                  # global average pooling instead of flatten
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc1 = nn.Linear(64 if gap else 64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def conv_block(self, x, conv, bn):
        x = conv(x)
        if bn is not None:
            x = bn(x)
        return self.pool(torch.relu(x))

    def forward(self, x):
        x = self.conv_block(x, self.conv1, self.bn1)    # 96 -> 48
        x = self.conv_block(x, self.conv2, self.bn2)    # 48 -> 24
        x = self.conv_block(x, self.conv3, self.bn3)    # 24 -> 12
        x = self.conv_block(x, self.conv4, self.bn4)    # 12 -> 6
        x = x.mean((2, 3)) if self.gap else x.flatten(1)
        x = torch.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)


def mobilenet(freeze_backbone):
    """The Task 2 network: a pretrained MobileNetV3-Small with a new last layer."""
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    return model


# ---------------------------------------------------------------------- transforms
NORMALIZE = transforms.Normalize(MEAN, STD)
plain = transforms.Compose([transforms.ToTensor(), NORMALIZE])
GEOMETRIC = [transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
             transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
             transforms.RandomRotation(20)]
COLOUR = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)


def train_transform(augment=()):
    """The Task 3 transform, with the requested augmentation in front of it."""
    return transforms.Compose(list(augment) + [transforms.ToTensor(), NORMALIZE])


# ----------------------------------------------------------------------- variants
def variant(**kw):
    base = dict(change="the Task 3 pipeline itself", net=dict(), augment=(), optimizer="adam",
                lr=1e-3, weight_decay=0.0, epochs=15, scheduler=None, label_smoothing=0.0,
                best_epoch=False, pretrained=False, freeze_backbone=False, historical=False)
    return {**base, **kw}


VARIANTS = {   # no commas in a name: --only and --save take comma-separated lists
    "baseline":        variant(),
    "+augmentation":   variant(change="geometric augmentation (Task 3d)", augment=GEOMETRIC),
    "+colour jitter":  variant(change="ColorJitter added to the augmentation", augment=[COLOUR] + GEOMETRIC),
    "+batchnorm":      variant(change="BatchNorm after every conv (Task 3d)", net=dict(bn=True)),
    "+dropout 0.3":    variant(change="dropout 0.3 after the first linear layer", net=dict(dropout=0.3)),
    "+global avg pool": variant(change="global average pooling instead of flatten", net=dict(gap=True)),
    "+adamw wd 1e-4":  variant(change="AdamW with weight decay 1e-4", optimizer="adamw", weight_decay=1e-4),
    "+label smoothing": variant(change="CrossEntropyLoss(label_smoothing=0.1)", label_smoothing=0.1),
    "+best epoch":     variant(change="keep the weights of the best validation epoch", best_epoch=True),
    "+cosine 30 ep":   variant(change="cosine LR over 30 epochs", epochs=30, scheduler="cosine"),
    "mobilenet frozen": variant(change="pretrained MobileNetV3, only the new last layer trained",
                                pretrained=True, freeze_backbone=True),
    "mobilenet tuned": variant(change="pretrained MobileNetV3, whole network fine-tuned",
                               pretrained=True),
    "mobilenet full":  variant(change="pretrained + augmentation + colour + cosine + 30 epochs",
                               pretrained=True, augment=[COLOUR] + GEOMETRIC, epochs=30, scheduler="cosine"),
    "cnn best":        variant(change="bn + gap + dropout + augmentation + colour + adamw + "
                                      "smoothing + cosine 30 ep + best epoch",
                               net=dict(bn=True, gap=True, dropout=0.3), augment=[COLOUR] + GEOMETRIC,
                               optimizer="adamw", weight_decay=1e-4, epochs=30, scheduler="cosine",
                               label_smoothing=0.1, best_epoch=True),
    # The combination the individual rows actually suggest: batchnorm, both augmentations, cosine
    # over 30 epochs and the best epoch kept. Without global average pooling (it loses 14 points on
    # the camera-like set), and without dropout, AdamW or label smoothing (all inside seed noise).
    "phone best":      variant(change="bn + augmentation + colour + cosine 30 ep + best epoch",
                               net=dict(bn=True), augment=[COLOUR] + GEOMETRIC,
                               epochs=30, scheduler="cosine", best_epoch=True),
    "cnn best +hist":  variant(change="the row above, last year's photos added to training",
                               net=dict(bn=True, gap=True, dropout=0.3), augment=[COLOUR] + GEOMETRIC,
                               optimizer="adamw", weight_decay=1e-4, epochs=30, scheduler="cosine",
                               label_smoothing=0.1, best_epoch=True, historical=True),
    "mobilenet +hist": variant(change="pretrained full, last year's photos added to training",
                               pretrained=True, augment=[COLOUR] + GEOMETRIC, epochs=30,
                               scheduler="cosine", best_epoch=True, historical=True),
}


# ------------------------------------------------------------------------ training
def accuracy(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def train(cfg, seed):
    torch.manual_seed(seed)

    train_transform_here = train_transform(cfg["augment"])
    train_sets = [datasets.ImageFolder(HERE / "data" / "leaves" / "train", transform=train_transform_here)]
    if cfg["historical"]:
        train_sets.append(datasets.ImageFolder(HERE / "data_history" / "leaves" / "train",
                                               transform=train_transform_here))
    # A seeded generator: every variant draws the same shuffled order of the same images.
    loader = DataLoader(ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0],
                        batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = (mobilenet(cfg["freeze_backbone"]) if cfg["pretrained"]
             else LeafNet(**cfg["net"])).to(device)
    params = [p for p in model.parameters() if p.requires_grad]      # the frozen head-only case
    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.Adam(params, lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
                 if cfg["scheduler"] == "cosine" else None)

    best_weights, best_val, last_loss = None, -1.0, float("nan")
    started = time.time()
    for epoch in range(cfg["epochs"]):
        if cfg["freeze_backbone"]:
            model.eval()                    # frozen BatchNorm must not re-estimate its statistics
            model.classifier.train()        # ...but the new head does train
        else:
            model.train()
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
        if scheduler is not None:
            scheduler.step()
        last_loss = loss_sum / len(loader.dataset)

        val = accuracy(model, VAL)
        if val > best_val:
            best_val, best_weights = val, copy.deepcopy(model.state_dict())

    if cfg["best_epoch"]:                   # the weights of the best epoch, not of the last one
        model.load_state_dict(best_weights)
    return dict(model=model, train_loss=last_loss, seconds=time.time() - started,
                parameters=sum(p.numel() for p in params),
                val=accuracy(model, VAL), camera=accuracy(model, CAMERA), others=accuracy(model, OTHERS),
                best_val=best_val)


# --------------------------------------------------------------------------- main
VAL = DataLoader(datasets.ImageFolder(HERE / "data" / "leaves" / "val", transform=plain), batch_size=BATCH)
CAMERA = DataLoader(datasets.ImageFolder(HERE / "data" / "leaves_camera" / "val", transform=plain),
                    batch_size=BATCH)
OTHERS = DataLoader(datasets.ImageFolder(HERE / "data_history" / "leaves" / "val", transform=plain),
                    batch_size=BATCH)


def mean(values):
    return sum(values) / len(values)


def mean_and_std(values):
    """Mean and sample standard deviation — with one run there is no spread to report."""
    return mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def append_run(row):
    new_file = not RUNS.exists()
    with open(RUNS, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, RUN_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_tables():
    """Average the recorded runs per variant into results_task4.csv / .json / .md."""
    with open(RUNS, encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    for run in runs:                                  # csv reads everything as text
        for column in RUN_COLUMNS[2:]:
            run[column] = float(run[column])

    summary = []
    for name in VARIANTS:                             # baseline first, then the order of the table
        these = [r for r in runs if r["variant"] == name]
        if not these:
            continue
        row = {"variant": name, "runs": len(these), "change": VARIANTS[name]["change"],
               "epochs": int(these[0]["epochs"]), "trainable_parameters": int(these[0]["parameters"]),
               "minutes_per_run": round(mean([r["seconds"] for r in these]) / 60, 2)}
        for metric in ["val", "camera", "others"]:
            row[metric], row[metric + "_std"] = [round(v, 4) for v in mean_and_std([r[metric] for r in these])]
        summary.append(row)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "summary": summary}, f, indent=1)

    table = ("| variant | val | camera-like | other phones | epochs | trainable parameters | minutes per run |\n"
             "|---|---|---|---|---|---|---|\n")
    for row in summary:
        cells = [f"{row[metric] * 100:.1f} ± {row[metric + '_std'] * 100:.1f}"
                 if row["runs"] > 1 else f"{row[metric] * 100:.1f}"
                 for metric in ["val", "camera", "others"]]
        table += (f"| {row['variant']} | {' | '.join(cells)} | {row['epochs']} "
                  f"| {row['trainable_parameters']:,} | {row['minutes_per_run']} |\n")
    SUMMARY_MD.write_text(table, encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(VARIANTS), help="comma-separated variant names")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    parser.add_argument("--save", default="", help="comma-separated variants whose weights to keep")
    args = parser.parse_args()
    chosen = [v.strip() for v in args.only.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"device {device} | val {len(VAL.dataset)} images | camera {len(CAMERA.dataset)} "
          f"| other phones {len(OTHERS.dataset)} | seeds {seeds}")
    keep = [v.strip() for v in args.save.split(",") if v.strip()]
    for name in chosen:
        cfg = VARIANTS[name]
        for seed in seeds:
            result = train(cfg, seed)
            row = {"variant": name, "seed": seed, "epochs": cfg["epochs"],
                   "parameters": result["parameters"], "val": result["val"],
                   "camera": result["camera"], "others": result["others"],
                   "best_val": result["best_val"], "train_loss": result["train_loss"],
                   "seconds": round(result["seconds"], 1)}
            if name in keep:                       # so export.py can export exactly this run
                weights = RUNS.parent / "weights" / f"{name.replace(' ', '_')}_seed{seed}.pt"
                weights.parent.mkdir(exist_ok=True)
                torch.save(result["model"].state_dict(), weights)
            append_run(row)
            print(f"{name:20s} seed {seed}  val {result['val']:.3f}  camera {result['camera']:.3f}  "
                  f"others {result['others']:.3f}  loss {result['train_loss']:.3f}  "
                  f"{result['seconds']:.0f}s")
    write_tables()
