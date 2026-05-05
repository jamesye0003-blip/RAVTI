#!/usr/bin/env python3
"""
Train a torchvision image classifier on iNaturalist for CAS@1 / CAS@5 evaluation.

Checkpoint format matches ``ClassificationAccuracyMetric.from_config`` (plain ``state_dict``
or dict with ``state_dict`` key). Label indices match torchvision ``INaturalist``
``target_type='full'`` category ids — the same numbering used by
``scripts/enrich_inat_checklist_class_index.py``.

Example::

  python scripts/train_cas_classifier.py --config configs/inaturalist.yaml \\
    --epochs 5 --batch-size 32 --output outputs/cas_classifier/inat_resnet50.pt

Then set ``evaluation.cas_classifier``, ``cas_classifier_arch``, ``cas_num_classes``
in YAML (num_classes printed at end).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import INaturalist

from ravti.config import load_yaml_config
from ravti.paths import project_root


class MappedSubset(torch.utils.data.Dataset):
    """Apply a transform to a torch Subset item (Windows multiprocessing-safe)."""

    def __init__(self, subset, tf, label_remap: dict[int, int] | None = None) -> None:
        self.subset = subset
        self.tf = tf
        self.label_remap = label_remap or {}

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, y = self.subset[idx]
        y_int = int(y)
        if self.label_remap:
            if y_int not in self.label_remap:
                raise KeyError(f"Label {y_int} missing in label_remap")
            y_int = int(self.label_remap[y_int])
        return self.tf(img), y_int


class ManifestClassificationDataset(torch.utils.data.Dataset):
    """Classification dataset from JSONL manifest with image_path + species/class_index."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        use_class_index: bool = True,
    ) -> None:
        self.rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    self.rows.append(dict(row))
        if not self.rows:
            raise ValueError(f"Manifest has no usable rows: {manifest_path}")
        self.use_class_index = bool(use_class_index)
        self.label_remap: dict[Any, int] = {}
        next_idx = 0
        for row in self.rows:
            key: Any
            if self.use_class_index and row.get("class_index") is not None:
                key = int(row["class_index"])
            else:
                key = str(row.get("species") or "").strip()
            if key not in self.label_remap:
                self.label_remap[key] = next_idx
                next_idx += 1
        self.num_classes = len(self.label_remap)
        if self.num_classes < 2:
            raise ValueError(f"Need at least 2 classes in manifest, got {self.num_classes}")
        self.class_index_remap: dict[int, int] = {}
        if self.use_class_index:
            for raw, mapped in self.label_remap.items():
                try:
                    self.class_index_remap[int(raw)] = int(mapped)
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.rows[idx]
        raw_path = row.get("image_path")
        if not raw_path:
            raise ValueError(f"Missing image_path in manifest row: {row}")
        p = Path(str(raw_path))
        img = Image.open(p).convert("RGB")
        if self.use_class_index and row.get("class_index") is not None:
            key = int(row["class_index"])
        else:
            key = str(row.get("species") or "").strip()
        y = int(self.label_remap[key])
        return img, y


def _load_checklist_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return [dict(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("checklist"), list):
        return [dict(x) for x in data["checklist"]]
    raise ValueError(f"Checklist must be a YAML list or mapping(checklist: ...), got {type(data)} from {path}")


def _extract_class_indices(rows: list[dict]) -> list[int]:
    out: list[int] = []
    for r in rows:
        raw = r.get("class_index", r.get("target_class_index"))
        if raw is None:
            continue
        try:
            out.append(int(raw))
        except Exception:
            continue
    out = sorted(set(out))
    if not out:
        raise ValueError("Checklist has no valid class_index values; cannot restrict CAS classes.")
    return out


def _inat_root(cfg: dict) -> Path:
    ds_cfg = cfg.get("dataset") or {}
    icfg = ds_cfg.get("inaturalist") or {}
    raw_root = icfg.get("root")
    if raw_root is None:
        return (project_root() / "data" / "datasets" / "inaturalist").resolve()
    root = Path(str(raw_root))
    if not root.is_absolute():
        root = (project_root() / root).resolve()
    return root


def _resolve_resnet50_weights(pretrained: bool):
    if not pretrained:
        return None
    try:
        from torchvision.models import ResNet50_Weights

        return ResNet50_Weights.IMAGENET1K_V1
    except Exception:
        return True  # older torchvision: resnet50(weights=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAS classifier on iNaturalist")
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "inaturalist.yaml"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Held-out fraction of train split")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50"])
    parser.add_argument("--pretrained-backbone", action="store_true", default=True)
    parser.add_argument("--no-pretrained-backbone", action="store_true", help="Train ResNet from scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100, help="Train progress print interval (steps)")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Stop if val_acc does not improve for this many epochs (<=0 disables early stop).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_acc improvement counted as progress for early stopping.",
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="Optional checklist YAML path for class subset training (uses class_index).",
    )
    parser.add_argument(
        "--restrict-to-checklist-classes",
        action="store_true",
        help="Train classifier only on classes present in checklist class_index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for .pt checkpoint (default: outputs/cas_classifier/resnet50_<version>.pt)",
    )
    parser.add_argument("--download", action="store_true", help="Allow iNaturalist download if missing")
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL manifest for classifier training (e.g., split eval_manifest).",
    )
    parser.add_argument(
        "--manifest-use-class-index",
        action="store_true",
        default=True,
        help="When using --manifest-jsonl, prefer row.class_index as label id before remap.",
    )
    args = parser.parse_args()

    pretrained = args.pretrained_backbone and not args.no_pretrained_backbone

    cfg = load_yaml_config(Path(args.config))
    ds_cfg = cfg.get("dataset") or {}
    icfg = ds_cfg.get("inaturalist") or {}
    version = str(icfg.get("version", "2021_train_mini"))
    root = _inat_root(cfg)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[init] device={device}  epochs={args.epochs}  batch_size={args.batch_size}  "
        f"num_workers={args.num_workers}  pretrained_backbone={pretrained}",
        flush=True,
    )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    global_to_local: dict[int, int] | None = None
    local_to_global: list[int] | None = None
    if args.manifest_jsonl is not None:
        manifest = args.manifest_jsonl
        if not manifest.is_absolute():
            manifest = (project_root() / manifest).resolve()
        print(f"[data] loading manifest dataset from {manifest} ...", flush=True)
        base = ManifestClassificationDataset(
            manifest,
            use_class_index=bool(args.manifest_use_class_index),
        )
        train_base: torch.utils.data.Dataset = base
        num_classes = int(base.num_classes)
        if base.class_index_remap:
            global_to_local = dict(base.class_index_remap)
            local_to_global = [0] * len(global_to_local)
            for g, l in global_to_local.items():
                if l >= len(local_to_global):
                    local_to_global.extend([0] * (l - len(local_to_global) + 1))
                local_to_global[l] = g
    else:
        print(f"[data] loading INaturalist version={version} from root={root} ...", flush=True)
        base = INaturalist(
            root=str(root),
            version=version,
            transform=None,
            target_type="full",
            download=bool(args.download or icfg.get("download", False)),
        )
        if not hasattr(base, "all_categories"):
            raise RuntimeError("This torchvision build's INaturalist lacks all_categories.")
        num_classes = len(base.all_categories)
        if num_classes < 2:
            raise RuntimeError(f"iNaturalist version={version} has unexpected class count={num_classes}")
        train_base = base
    if args.restrict_to_checklist_classes and args.manifest_jsonl is None:
        ev = cfg.get("evaluation") or {}
        gen = ev.get("generation") or {}
        checklist_path = args.checklist
        if checklist_path is None:
            raw = gen.get("prompts_file")
            if not raw:
                raise ValueError(
                    "--restrict-to-checklist-classes requires --checklist or evaluation.generation.prompts_file in config."
                )
            checklist_path = Path(str(raw))
        if not checklist_path.is_absolute():
            checklist_path = (project_root() / checklist_path).resolve()
        rows = _load_checklist_rows(checklist_path)
        local_to_global = _extract_class_indices(rows)
        global_to_local = {gid: i for i, gid in enumerate(local_to_global)}
        selected_indices = [i for i, (cat_id, _fname) in enumerate(base.index) if int(cat_id) in global_to_local]
        if not selected_indices:
            raise ValueError(
                f"No samples matched checklist classes ({len(local_to_global)} ids) in dataset {version}."
            )
        train_base = Subset(base, selected_indices)
        num_classes = len(local_to_global)
        print(
            f"[data] class subset enabled from checklist={checklist_path} "
            f"global_classes={len(local_to_global)} matched_samples={len(selected_indices)}",
            flush=True,
        )

    n_total = len(train_base)
    n_val = max(1, int(round(n_total * args.val_fraction)))
    n_train = n_total - n_val
    if n_train < 1:
        raise RuntimeError(f"Train split empty (n_total={n_total}, val_fraction={args.val_fraction})")
    print(
        f"[data] loaded samples={n_total}  classes={num_classes}  train={n_train}  val={n_val}",
        flush=True,
    )

    train_ds, val_ds = random_split(train_base, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    if args.manifest_jsonl is not None:
        train_final = MappedSubset(Subset(train_base, train_ds.indices), train_tf, label_remap=None)
        val_final = MappedSubset(Subset(train_base, val_ds.indices), val_tf, label_remap=None)
    else:
        train_final = MappedSubset(train_ds, train_tf, label_remap=global_to_local)
        val_final = MappedSubset(val_ds, val_tf, label_remap=global_to_local)

    train_loader = DataLoader(
        train_final,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_final,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    import torchvision.models as tvm

    print(f"[model] building {args.arch} ...", flush=True)
    weights = _resolve_resnet50_weights(pretrained)
    if args.arch != "resnet50":
        raise ValueError(f"Unsupported arch={args.arch}")
    model = tvm.resnet50(weights=weights)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)

    model = model.to(device)
    print("[model] ready; start training", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    patience = int(args.early_stop_patience)
    min_delta = float(args.early_stop_min_delta)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"train ep{epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for step, (xb, yb) in enumerate(train_pbar, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
            if step % max(1, args.log_interval) == 0:
                train_pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
        train_loss = running / max(n_seen, 1)

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            val_pbar = tqdm(val_loader, desc=f"val   ep{epoch + 1}/{args.epochs}", dynamic_ncols=True, leave=False)
            for xb, yb in val_pbar:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total += yb.size(0)
        val_acc = correct / max(total, 1)

        print(f"[epoch] {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
            print(f"[early-stop] improvement accepted: best_val_acc={best_val_acc:.4f}", flush=True)
        else:
            stale_epochs += 1
            if patience > 0:
                print(
                    f"[early-stop] no significant improvement for {stale_epochs}/{patience} epoch(s)",
                    flush=True,
                )
                if stale_epochs >= patience:
                    print(
                        f"[early-stop] triggered at epoch {epoch + 1}/{args.epochs}; "
                        f"best_val_acc={best_val_acc:.4f}",
                        flush=True,
                    )
                    break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out = args.output
    if out is None:
        safe_ver = version.replace("/", "_")
        out = project_root() / "outputs" / "cas_classifier" / f"{args.arch}_{safe_ver}.pt"
    else:
        if not out.is_absolute():
            out = (project_root() / out).resolve()

    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": best_state,
        "num_classes": num_classes,
        "arch": args.arch,
        "inat_version": version,
        "inat_root": str(root.resolve()),
        "val_accuracy": float(best_val_acc),
        "eval_note": "Use with evaluation.cas_classifier + cas_num_classes=num_classes above; "
        "cas_classifier_arch must match arch.",
    }
    if global_to_local is not None and local_to_global is not None:
        payload["class_index_remap"] = global_to_local  # global class_index -> local classifier index
        payload["class_index_vocab"] = local_to_global  # local index -> global class_index
    if args.manifest_jsonl is not None:
        payload["manifest_jsonl"] = str(args.manifest_jsonl)
    torch.save(payload, out)

    meta_path = out.with_suffix(".json")
    meta_path.write_text(
        json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {"checkpoint": str(out), "meta": str(meta_path), **{k: v for k, v in payload.items() if k != "state_dict"}},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
