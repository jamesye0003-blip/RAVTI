#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ravti.config import load_yaml_config
from ravti.eval.species_checklist import common_name_from_meta, taxonomy_line_from_meta
from ravti.eval.t2i_alignment import build_t2i_metrics_from_config
from ravti.paths import project_root


def _safe_image_stem(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        s = "sample"
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = "_".join(s.split())
    return s[:180]


def _strip_double_ext_stem(name: str) -> str:
    s = Path(name).name
    for _ in range(4):
        t = Path(s).stem
        if t == s:
            break
        s = t
    return s


def _provider_key(raw: str) -> str:
    k = str(raw).strip().lower()
    if k in {"inat", "inaturalist", "inaturalist_mini", "inat_mini", "inat2021_mini"}:
        return "inaturalist_mini"
    if k in {"fishnet"}:
        return "fishnet"
    raise ValueError(f"Unknown provider={raw!r}; use one of [inat, inaturalist_mini, fishnet].")


def _provider_selected_dir(provider: str) -> str:
    return "inat" if provider == "inaturalist_mini" else "fishnet"


def _load_prepare_module():
    root = project_root()
    path = root / "scripts" / "prepare_dataset_pipeline.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assign_class_index(records: list[dict[str, Any]]) -> dict[str, int]:
    species_sorted = sorted({str(r["species"]).strip() for r in records})
    mapping = {sp: i for i, sp in enumerate(species_sorted)}
    for r in records:
        r["class_index"] = mapping[str(r["species"]).strip()]
    return mapping


def _count_bucket(n_images: int) -> str:
    n = int(n_images)
    if n <= 2:
        return "2"
    if n <= 5:
        return "3_5"
    if n <= 10:
        return "6_10"
    return "11_plus"


def _species_count_map(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in records:
        out[str(r["species"]).strip()] += 1
    return dict(out)


def _build_eval_stem_map(eval_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in eval_rows:
        sid = str(row.get("sample_id", "")).strip()
        stem = _safe_image_stem(sid)
        # sample_id 可能带文件后缀（如 fishnet_xxx.jpg），而生成图名常是 fishnet_xxx.jpg.png；
        # 这里同时保存“原始 stem”和“去双后缀 stem”两种键，提升回溯鲁棒性。
        out[stem] = dict(row)
        out[_strip_double_ext_stem(stem)] = dict(row)
    return out


def _resolve_default_manifests(cfg: dict[str, Any]) -> tuple[Path, Path]:
    root = project_root()
    ds_split = ((cfg.get("dataset") or {}).get("split") or {})
    ev_gen = ((cfg.get("evaluation") or {}).get("generation") or {})
    eval_raw = ev_gen.get("eval_manifest_file") or ds_split.get("eval_manifest_jsonl")
    if not eval_raw:
        raise ValueError("Cannot infer eval manifest from config.")
    eval_path = Path(str(eval_raw))
    if not eval_path.is_absolute():
        eval_path = (root / eval_path).resolve()
    # 优先与当前 eval_manifest 同目录，确保与选图来源 split 一致。
    all_path = (eval_path.parent / "all_manifest.jsonl").resolve()
    if not all_path.is_file():
        train_raw = ds_split.get("train_manifest_jsonl")
        if train_raw:
            train_path = Path(str(train_raw))
            if not train_path.is_absolute():
                train_path = (root / train_path).resolve()
            all_path = (train_path.parent / "all_manifest.jsonl").resolve()
    return eval_path, all_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 selected_imgs/{provider} 中 base 图自动回溯并用 CLIP/BioCLIP 选 top-k 物种，输出极小 split。"
    )
    parser.add_argument("--config", type=str, required=True, help="对应 provider 的 yaml（读取模型与默认 manifest）")
    parser.add_argument("--provider", type=str, required=True, help="inat|inaturalist_mini|fishnet")
    parser.add_argument("--k", type=int, required=True, help="最终保留的物种数（按 score 排序后 top-k）")
    parser.add_argument("--split-name", type=str, required=True, help="输出 split 目录名")
    parser.add_argument(
        "--score-metric",
        type=str,
        default="bioclip_taxonomic",
        choices=["clip_common_name", "bioclip_taxonomic", "hybrid"],
        help="评分：CLIP(common_name) / BioCLIP(taxonomy_line) / 二者平均",
    )
    parser.add_argument("--selected-root", type=str, default="outputs/selected_imgs")
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument("--all-manifest", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-per-species", type=int, default=1)
    parser.add_argument("--max-train-per-species", type=int, default=0)
    parser.add_argument("--min-samples-per-species", type=int, default=2)
    parser.add_argument("--exposed", action="store_true")
    args = parser.parse_args()

    root = project_root()
    provider = _provider_key(args.provider)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = load_yaml_config(cfg_path)

    selected_root = Path(args.selected_root)
    if not selected_root.is_absolute():
        selected_root = (root / selected_root).resolve()
    selected_dir = selected_root / _provider_selected_dir(provider)
    if not selected_dir.is_dir():
        raise SystemExit(f"Selected images directory not found: {selected_dir}")

    if args.eval_manifest:
        eval_path = Path(args.eval_manifest)
        if not eval_path.is_absolute():
            eval_path = (root / eval_path).resolve()
    else:
        eval_path, _ = _resolve_default_manifests(cfg)
    if args.all_manifest:
        all_path = Path(args.all_manifest)
        if not all_path.is_absolute():
            all_path = (root / all_path).resolve()
    else:
        _, all_path = _resolve_default_manifests(cfg)

    if not eval_path.is_file():
        raise SystemExit(f"eval manifest not found: {eval_path}")
    if not all_path.is_file():
        raise SystemExit(f"all manifest not found: {all_path}")

    eval_rows = _read_jsonl(eval_path)
    all_rows = _read_jsonl(all_path)
    stem_map = _build_eval_stem_map(eval_rows)

    clip_metric, bioclip_metric = build_t2i_metrics_from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = sorted(
        p for p in selected_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not images:
        raise SystemExit(f"No selected images found in {selected_dir}")

    scored_rows: list[dict[str, Any]] = []
    for p in images:
        stem = _strip_double_ext_stem(p.name)
        row = stem_map.get(stem)
        if row is None:
            print(f"WARN: no eval row for {p.name} (stem={stem})")
            continue
        species = str(row.get("species") or "").strip()
        if not species:
            continue
        taxon_line = taxonomy_line_from_meta(row, species)
        common_name = common_name_from_meta(row, species)
        img = Image.open(p).convert("RGB")
        clip_s = float(clip_metric.score(img, common_name, device))
        bio_s = float(bioclip_metric.score(img, taxon_line, device))
        if args.score_metric == "clip_common_name":
            score = clip_s
        elif args.score_metric == "bioclip_taxonomic":
            score = bio_s
        else:
            score = 0.5 * (clip_s + bio_s)
        scored_rows.append(
            {
                "species": species,
                "score": float(score),
                "clip_common_name_score": float(clip_s),
                "bioclip_taxonomic_score": float(bio_s),
                "common_name": common_name,
                "taxonomy_line": taxon_line,
                "sample_stem": stem,
                "gen_image": str(p.resolve()),
            }
        )

    if not scored_rows:
        raise SystemExit("No valid selected images could be mapped and scored.")

    best_by_species: dict[str, dict[str, Any]] = {}
    for rec in scored_rows:
        sp = str(rec["species"])
        prev = best_by_species.get(sp)
        if prev is None or float(rec["score"]) > float(prev["score"]):
            best_by_species[sp] = rec

    ranked = sorted(best_by_species.values(), key=lambda x: -float(x["score"]))
    k = int(args.k)
    if k > len(ranked):
        raise SystemExit(f"k={k}, but only {len(ranked)} distinct species are available.")
    chosen = ranked[:k]
    chosen_species = [str(x["species"]).strip() for x in chosen]
    chosen_set = set(chosen_species)

    subset_records = [r for r in all_rows if str(r.get("species", "")).strip() in chosen_set]
    if not subset_records:
        raise SystemExit("No rows in all_manifest matched selected species.")

    prep = _load_prepare_module()
    train_rows, eval_split_rows, split_stats = prep._split_records(
        subset_records,
        eval_per_species=int(args.eval_per_species),
        max_train_per_species=int(args.max_train_per_species),
        min_samples_per_species=int(args.min_samples_per_species),
        exposed=bool(args.exposed),
        seed=int(args.seed),
    )

    species_to_class = _assign_class_index(subset_records)
    subset_counts = _species_count_map(subset_records)
    bucket_hist: dict[str, int] = {"2": 0, "3_5": 0, "6_10": 0, "11_plus": 0}
    for sp in chosen_species:
        bucket_hist[_count_bucket(int(subset_counts.get(sp, 0)))] += 1

    out_dir = root / "data" / "metadata" / "splits" / provider / str(args.split_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_out = out_dir / "all_manifest.jsonl"
    train_out = out_dir / "train_manifest.jsonl"
    eval_out = out_dir / "eval_manifest.jsonl"
    class_map_out = out_dir / "class_index_map.json"
    subset_species_out = out_dir / "subset_species.txt"
    subset_species_json = out_dir / "subset_species.json"
    ranking_out = out_dir / "baseline_curated_ranking.json"
    meta_out = out_dir / "split_meta.json"

    _write_jsonl(all_out, subset_records)
    _write_jsonl(train_out, train_rows)
    _write_jsonl(eval_out, eval_split_rows)
    class_map_out.write_text(json.dumps(species_to_class, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subset_species_out.write_text("\n".join(chosen_species) + ("\n" if chosen_species else ""), encoding="utf-8")
    subset_species_json.write_text(
        json.dumps(
            [{"species": sp, "n_images": int(subset_counts.get(sp, 0))} for sp in chosen_species],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ranking_out.write_text(
        json.dumps(
            {
                "provider": provider,
                "score_metric": args.score_metric,
                "selected_images_dir": str(selected_dir),
                "k_requested": k,
                "n_scored_rows": len(scored_rows),
                "ranked_species": chosen,
                "eval_manifest": str(eval_path),
                "all_manifest_source": str(all_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "script": "build_curated_baseline_subset.py",
        "provider": provider,
        "split_name": str(args.split_name),
        "score_metric": args.score_metric,
        "selected_images_dir": str(selected_dir),
        "all_manifest_jsonl": str(all_out),
        "train_manifest_jsonl": str(train_out),
        "eval_manifest_jsonl": str(eval_out),
        "class_index_map_json": str(class_map_out),
        "subset_species_txt": str(subset_species_out),
        "subset_species_json": str(subset_species_json),
        "ranking_json": str(ranking_out),
        "subset_enabled": True,
        "subset_num_species_requested": int(k),
        "subset_num_species_selected": len(chosen_species),
        "subset_min_images_per_species": int(args.min_samples_per_species),
        "subset_bucket_hist_species": bucket_hist,
        "curated_species": chosen_species,
        "n_selected_images_input": len(images),
        "n_scored_images": len(scored_rows),
        **split_stats,
    }
    meta_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
