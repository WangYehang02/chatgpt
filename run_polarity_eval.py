#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四数据集无监督极性策略对比：auto_vote（多探针投票） vs legacy_lcc（单 LCC-Spearman 阈值）。
每数据集每 seed 各跑两次 main_train，收集 AUC/AP 与极性诊断；汇总到 results/polarity_eval_summary.md
"""
import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

FMGAD_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIGS = {
    "enron": "enron_best.yaml",
    "disney": "disney_best.yaml",
    "reddit": "reddit_best.yaml",
    "books": "books_best.yaml",
}

LEGACY_OVERRIDES: Dict[str, Any] = {
    "polarity_mode": "legacy_lcc",
    "lcc_spearman_polarity": False,  # 由 polarity_mode 显式指定
}


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.Loader)


def write_yaml(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _extract_rhoes(di: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    r_lcc, r_deg = None, None
    if "rho_lcc" in di and di["rho_lcc"] is not None:
        r_lcc = float(di["rho_lcc"])
    for p in di.get("probes", []) or []:
        n = p.get("name", "")
        if n == "lcc_spearman" and p.get("rho") is not None:
            r_lcc = r_lcc if r_lcc is not None else float(p["rho"])
        if n == "deg_spearman" and p.get("rho") is not None:
            r_deg = float(p["rho"])
    return r_lcc, r_deg


def one_run(
    config_path: str,
    seed: int,
    device: int,
    out_json: str,
    main_py: str,
) -> None:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    cmd = [
        sys.executable,
        main_py,
        "--config",
        config_path,
        "--seed",
        str(seed),
        "--device",
        str(device),
        "--result-file",
        out_json,
    ]
    env = os.environ.copy()
    t0 = time.time()
    subprocess.run(cmd, cwd=FMGAD_ROOT, check=True, env=env)
    with open(out_json, "r", encoding="utf-8") as f:
        p = json.load(f)
    p["wall_time_sec"] = time.time() - t0
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


def aggregate_by_dataset(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        key = (r["dataset"], r["mode"])
        by.setdefault(key, []).append(r)
    out = {}
    for (dset, mode), rows in by.items():
        aucs = [float(x["auc_mean"]) for x in rows]
        aps = [float(x.get("ap_mean", 0.0)) for x in rows]
        flips = []
        for x in rows:
            pd = x.get("polarity_diagnostics") or {}
            fl = pd.get("flipped")
            if fl is not None and isinstance(fl, bool):
                flips.append(1 if fl else 0)
        out[(dset, mode)] = {
            "n": len(aucs),
            "auc_mean": statistics.mean(aucs) if aucs else float("nan"),
            "auc_std": statistics.pstdev(aucs) if len(aucs) > 1 else 0.0,
            "ap_mean": statistics.mean(aps) if aps else float("nan"),
            "ap_std": statistics.pstdev(aps) if len(aps) > 1 else 0.0,
            "flip_rate": (sum(flips) / len(flips)) if flips else 0.0,
            "n_flip_known": len(flips),
        }
    return out


def write_markdown(
    path: str,
    seeds: List[int],
    records: List[Dict[str, Any]],
) -> None:
    agg = aggregate_by_dataset(records)
    dsets = sorted({r["dataset"] for r in records})
    lines: List[str] = []
    lines.append("# 无监督极性评估 — auto_vote vs legacy_lcc\n\n")
    lines.append(f"Seeds: {seeds}。本表由 `run_polarity_eval.py` 在本地训练运行后自动写入。\n\n")

    lines.append("## 每数据集各策略 mean±std AUC / AP\n\n")
    lines.append("| dataset | auto_vote AUC | legacy_lcc AUC | auto_vote AP | legacy_lcc AP |\n")
    lines.append("|---------|--------------|----------------|--------------|---------------|\n")
    for d in dsets:
        a = agg.get((d, "auto_vote"), {})
        l = agg.get((d, "legacy_lcc"), {})
        if a and l:
            lines.append(
                f"| {d} | {a.get('auc_mean', 0):.4f}±{a.get('auc_std', 0):.4f} | "
                f"{l.get('auc_mean', 0):.4f}±{l.get('auc_std', 0):.4f} | "
                f"{a.get('ap_mean', 0):.4f}±{a.get('ap_std', 0):.4f} | "
                f"{l.get('ap_mean', 0):.4f}±{l.get('ap_std', 0):.4f} |\n"
            )
        elif a:
            lines.append(
                f"| {d} | {a.get('auc_mean', 0):.4f}±{a.get('auc_std', 0):.4f} | - | "
                f"{a.get('ap_mean', 0):.4f}±{a.get('ap_std', 0):.4f} | - |\n"
            )
        elif l:
            lines.append(
                f"| {d} | - | {l.get('auc_mean', 0):.4f}±{l.get('auc_std', 0):.4f} | - | "
                f"{l.get('ap_mean', 0):.4f}±{l.get('ap_std', 0):.4f} |\n"
            )
    lines.append("\n## 极性翻转触发率（有诊断的 run）\n\n")
    lines.append("| dataset | auto_vote | legacy_lcc |\n")
    lines.append("|---------|----------|------------|\n")
    for d in dsets:
        a = agg.get((d, "auto_vote"), {})
        l = agg.get((d, "legacy_lcc"), {})
        lines.append(
            f"| {d} | {a.get('flip_rate', 0.0) * 100.0:.1f}% (n={a.get('n_flip_known', 0)}) | "
            f"{l.get('flip_rate', 0.0) * 100.0:.1f}% (n={l.get('n_flip_known', 0)}) |\n"
        )

    lines.append("\n## 逐次运行明细（含 rho、是否翻转）\n\n")
    for r in sorted(records, key=lambda x: (x["dataset"], x["mode"], x["seed"])):
        di = r.get("polarity_diagnostics") or {}
        r1, r2 = _extract_rhoes(di)
        lines.append(
            f"- **{r['dataset']}** seed={r['seed']} mode=**{r['mode']}**  "
            f"AUC={r.get('auc_mean', 0):.4f} AP={r.get('ap_mean', 0):.4f}  "
            f"flipped={di.get('flipped', None)}  "
            f"rho_lcc={r1} rho_deg={r2}\n"
        )

    lines.append("\n---\n*说明：legacy 行对应 `polarity_mode: legacy_lcc` 与 LCC-Spearman 单阈值；auto 行对应当前 `*_best.yaml` 的 `auto_vote`。*\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Run polarity ablation (auto_vote vs legacy_lcc) for four datasets.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--seeds",
        type=str,
        default="42,3407,2026",
        help="逗号分隔 seeds，默认 42,3407,2026",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(FMGAD_ROOT, "results"),
        help="JSON 子目录与 markdown 根目录",
    )
    p.add_argument("--skip-runs", action="store_true", help="只根据已有 json 重生成 summary，不训练")
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_CONFIGS.keys()),
        help="逗号分隔，默认四数据集全跑",
    )
    args = p.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dsets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    main_py = os.path.join(FMGAD_ROOT, "main_train.py")
    res_dir = os.path.join(args.out_dir, "polarity_eval_runs")
    os.makedirs(res_dir, exist_ok=True)
    sum_path = os.path.join(args.out_dir, "polarity_eval_summary.md")

    records: List[Dict[str, Any]] = []

    if not args.skip_runs:
        for dname in dsets:
            cfile = DATASET_CONFIGS.get(dname)
            if cfile is None:
                print(f"skip unknown dataset: {dname}", flush=True)
                continue
            base_path = os.path.join(FMGAD_ROOT, "configs", cfile)
            base = load_yaml(base_path)

            for mode in ("auto_vote", "legacy_lcc"):
                for seed in seeds:
                    cfg = copy.deepcopy(base)
                    tag = f"{dname}_seed{seed}_{mode}"
                    if mode == "legacy_lcc":
                        cfg.update(LEGACY_OVERRIDES)
                    tdir = tempfile.mkdtemp(prefix=f"pol_{tag}_", dir=res_dir)
                    tmp_config = os.path.join(tdir, "config.yaml")
                    write_yaml(cfg, tmp_config)
                    out_json = os.path.join(tdir, "out.json")
                    try:
                        print("===", tag, flush=True)
                        one_run(tmp_config, seed, args.device, out_json, main_py)
                        with open(out_json, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        pol = dict(payload.get("polarity_diagnostics", {}) or {})
                        if pol:
                            r1, r2 = _extract_rhoes(pol)
                        else:
                            r1, r2 = None, None
                        records.append(
                            {
                                "dataset": dname,
                                "mode": mode,
                                "seed": seed,
                                "auc_mean": float(payload.get("auc_mean", 0.0)),
                                "ap_mean": float(payload.get("ap_mean", 0.0)),
                                "polarity_diagnostics": pol,
                                "rho_lcc": r1,
                                "rho_deg": r2,
                            }
                        )
                    except Exception as e:
                        print(f"ERROR {tag}: {e}", flush=True)
                        records.append(
                            {
                                "dataset": dname,
                                "mode": mode,
                                "seed": seed,
                                "error": str(e),
                            }
                        )
        # 持久化原始记录
        with open(os.path.join(res_dir, "all_records.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    else:
        pjson = os.path.join(res_dir, "all_records.json")
        if os.path.isfile(pjson):
            with open(pjson, "r", encoding="utf-8") as f:
                records = json.load(f)

    if records:
        good = [r for r in records if "error" not in r]
        write_markdown(sum_path, seeds, good)
        n_err = len(records) - len(good)
        if n_err:
            with open(sum_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n*注意：{n_err} 次 run 因异常未写入上表。见控制台日志。*\n")
    else:
        os.makedirs(os.path.dirname(sum_path) or ".", exist_ok=True)
        with open(sum_path, "w", encoding="utf-8") as f:
            f.write(
                "# 极性评估摘要\n\n尚未有运行结果。"
                "请执行: `python run_polarity_eval.py --device 0`（会调用四数据集 x 2 策略 x 3 seeds，训练耗时较长）。\n"
                "若仅重排已有 `results/polarity_eval_runs/all_records.json`：`--skip-runs`。\n"
            )
    print("Wrote", sum_path, flush=True)


if __name__ == "__main__":
    main()
