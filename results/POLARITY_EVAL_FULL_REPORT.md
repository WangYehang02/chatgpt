# 无监督极性策略评估 — 全结果汇总报告

> **主题**：`auto_vote`（多探针投票）与 `legacy_lcc`（LCC-Spearman 单阈值）在 **enron / disney / reddit / books** 四数据集上的对比。  
> **Seeds**：42, 3407, 2026（每数据集 ×2 套极性策略，共 24 次独立训练与评估）。  
> **极性逻辑**：仅使用 `score` 与图统计量（LCC、度、子图边密度等），**标签 `y` 不参与**极性决策；AUC/AP 仅用于**事后**评估。

---

## 1. 实验设置

| 项 | 说明 |
|----|------|
| 代码入口 | `run_polarity_eval.py` |
| 基线 | `polarity_mode: legacy_lcc`（覆盖为与 LCC-Spearman 单阈值一致） |
| 新方法 | 各 `configs/*_best.yaml` 中 `polarity_mode: auto_vote` 及配套超参 |
| 训练 / 解释器 | 需带 `torch`、pygod 等（例：`conda env fmgad`） |
| 原始产物 | `polarity_eval_runs/all_records.json`、各 `*_seed*_{mode}.json` |
| 自动表 | `polarity_eval_summary.md`、`polarity_report_closure.md` |

**复现（断点续跑会跳过已有 `*_*.json`）：**

```bash
cd /path/to/FMGAD
/home/yehang/miniconda3/envs/fmgad/bin/python run_polarity_eval.py \
  --device 0 \
  --python /home/yehang/miniconda3/envs/fmgad/bin/python
```

若系统 `python3` 无 `tqdm`/`torch`，**必须**通过 `--python` 指定 Conda 环境；否则子进程会失败。

---

## 2. 执行状态摘要

- **首败（环境）**：使用无依赖的 `python3` 时出现 `ModuleNotFoundError: tqdm`（子进程为系统解释器）。  
- **成功跑通**：使用 `fmgad` 环境解释器后，**24/24 次**完成，并生成下述统计与逐 seed 结果。

---

## 3. 总览：每数据集 AUC / AP 与极性诊断

### 3.1 均值与标准差（AUC & AP）

| dataset | auto_vote AUC | legacy_lcc AUC | auto_vote AP | legacy_lcc AP |
|---------|--------------|----------------|--------------|---------------|
| enron   | 0.8111±0.0449 | 0.8223±0.0514 | 0.0104±0.0095 | 0.0111±0.0108 |
| reddit  | 0.5626±0.0232 | 0.5082±0.0267 | 0.0379±0.0031 | 0.0313±0.0017 |
| disney  | 0.3964±0.0939 | 0.6671±0.1422 | 0.0444±0.0054 | 0.1018±0.0326 |
| books   | 0.4271±0.0134 | 0.6071±0.0173 | 0.0188±0.0022 | 0.0292±0.0011 |

### 3.2 ΔAUC、AUC&lt;0.5 计数、auto 是否触发翻转（`flipped` 诊断）

| dataset | legacy AUC mean±std | auto AUC mean±std | ΔAUC (auto−legacy) | legacy 中 AUC&lt;0.5 的 seed 数 | auto 中 AUC&lt;0.5 的 seed 数 | auto 运行中 flipped 比例 |
|---------|--------------------|------------------|--------------------|------------------|-----------------|------------------------|
| enron   | 0.8223±0.0514      | 0.8111±0.0449    | −0.0112            | 0                 | 0               | 100%（3/3 次诊断为 True） |
| reddit  | 0.5082±0.0267      | 0.5626±0.0232    | **+0.0543**         | 1                 | 0               | 100%（3/3）            |
| disney  | 0.6671±0.1422      | 0.3964±0.0939    | −0.2707            | 1                 | 2*              | 0%（3/3 均为 False）   |
| books   | 0.6071±0.0173      | 0.4271±0.0134    | −0.1800            | 0                 | 3               | 0%（3/3 均为 False）   |

\* disney 的 **seed=2026** 在 auto 下 **AUC=0.5000**（等于 0.5，**不**计入「&lt;0.5」）。若将「=0.5」也视为需关注，可单独记为第 3 个边界点。

### 3.3 各策略下「极性是否触发翻转」的统计（来自诊断字段 `flipped`）

| dataset | auto_vote 中 flipped 为 True 的比例 | legacy_lcc 中 flipped 为 True 的比例 |
|---------|--------------------------------------|--------------------------------------|
| enron   | 100% (3/3)                           | 100% (3/3)                           |
| reddit  | 100% (3/3)                           | 0% (0/3)                              |
| disney  | 0% (0/3)                              | 66.7% (2/3)                          |
| books   | 0% (0/3)                              | 100% (3/3)                            |

**说明**：`flipped` 在 **auto** 下表示多探针投票**最终是否执行了分数线性翻转**；在 **legacy** 下表示 LCC-Spearman 单阈值**是否超过阈值而翻转**。**与 AUC 高低无简单一一对应**（见第 5 节解读）。

---

## 4. 全量 Seed 级明细

以下按 **dataset → seed** 排列；每格为同一训练下 **auto_vote** 与 **legacy_lcc** 的成对结果。

| dataset | seed | mode | AUC | AP | flipped | rho_lcc | rho_deg | flip_v | keep_v | sum_conf |
|---------|------|------|-----|-----|--------|--------|--------|--------|--------|----------|
| books | 42 | auto_vote | 0.4385 | 0.0213 | False | −0.0528 | 0.5583 | 1 | 1 | 0.6631 |
| books | 42 | legacy_lcc | 0.6216 | 0.0292 | True | −0.0646 |  |  |  |  |
| books | 2026 | auto_vote | 0.4083 | 0.0159 | False | −0.1501 | 0.6884 | 1 | 1 | 0.9355 |
| books | 2026 | legacy_lcc | 0.6170 | 0.0306 | True | −0.1501 |  |  |  |  |
| books | 3407 | auto_vote | 0.4345 | 0.0192 | False | −0.0729 | 0.5429 | 1 | 1 | 0.6491 |
| books | 3407 | legacy_lcc | 0.5828 | 0.0279 | True | −0.0859 |  |  |  |  |
| disney | 42 | auto_vote | 0.2726 | 0.0368 | False | −0.1312 | 0.7339 | 1 | 2 | 1.7818 |
| disney | 42 | legacy_lcc | 0.4661 | 0.0558 | False | 0.0634 |  |  |  |  |
| disney | 2026 | auto_vote | 0.5000 | 0.0484 | False |  |  | 0 | 1 | 0.6137 |
| disney | 2026 | legacy_lcc | 0.7726 | 0.1257 | True | −0.1277 |  |  |  |  |
| disney | 3407 | auto_vote | 0.4167 | 0.0481 | False | 0.0458 | 0.2807 | 1 | 1 | 0.4704 |
| disney | 3407 | legacy_lcc | 0.7627 | 0.1240 | True | −0.1138 |  |  |  |  |
| enron | 42 | auto_vote | 0.8526 | 0.0060 | True | −0.0685 | −0.1318 | 2 | 0 | 0.2255 |
| enron | 42 | legacy_lcc | 0.8515 | 0.0046 | True | −0.0644 |  |  |  |  |
| enron | 2026 | auto_vote | 0.7487 | 0.0235 | True | −0.1029 | 0.0277 | 1 | 0 | 0.1805 |
| enron | 2026 | legacy_lcc | 0.7500 | 0.0263 | True | −0.1022 |  |  |  |  |
| enron | 3407 | auto_vote | 0.8321 | 0.0016 | True | −0.0717 | −0.1197 | 2 | 0 | 0.2191 |
| enron | 3407 | legacy_lcc | 0.8653 | 0.0024 | True | −0.0848 |  |  |  |  |
| reddit | 42 | auto_vote | 0.5930 | 0.0423 | True |  | −0.4864 | 1 | 0 | 0.4891 |
| reddit | 42 | legacy_lcc | 0.4754 | 0.0293 | False |  |  |  |  |  |
| reddit | 2026 | auto_vote | 0.5368 | 0.0351 | True |  | −0.5655 | 1 | 0 | 0.5680 |
| reddit | 2026 | legacy_lcc | 0.5409 | 0.0334 | False |  |  |  |  |  |
| reddit | 3407 | auto_vote | 0.5579 | 0.0364 | True |  | −0.6006 | 1 | 0 | 0.6032 |
| reddit | 3407 | legacy_lcc | 0.5084 | 0.0311 | False |  |  |  |  |  |

*注：`rho_lcc` / `rho_deg` 在 reddit 部分行为空，多因 LCC/分数常数等导致 Spearman 未定义或诊断以度探针为主；`flip_v` / `keep_v` / `sum_conf` 为 auto 策略专属。*

### 4.1 逐条探针票型（`auto_vote`，摘自自动日志）

- **books**：三 seed 均为 `lcc_spearman:flip, deg_spearman:keep, topq_density:abstain`，`fv/kv/m=1/1/1` → **未达「flip ≥ keep+margin」**，故 **flipped=False**，但 **sum_confidence** 较高（~0.65–0.94），属「多票相抵、不翻」。
- **disney**：`seed=42` 有 **keep=2**；`2026` 多 **abstain**；`3407` 出现 **topq 投 flip** 等组合，**均未最终翻转**（flipped 全 false）。
- **enron / reddit（auto）**：多 seed 满足 `flip_votes` 相对 `keep_votes` 优势 + 置信度门槛，**flipped=True**；reddit 上 LCC 常 **abstain**、**deg 投 flip** 仍触发翻转（见上表）。

---

## 5. 结果解读与结论

### 5.1 按数据集简要结论

| 数据集 | 结论（基于本批 3 seeds） |
|--------|--------------------------|
| **enron** | 两策略 AUC 均高（~0.75–0.86），差异小；**auto/legacy 诊断均常翻转**（flipped 全 True，legacy 3/3 亦为 True），属「都认可需对齐方向」之情形。 |
| **reddit** | **auto 平均 AUC 与 AP 均优于 legacy**；**legacy 有 1 个 seed（42）AUC&lt;0.5**，**auto 三 seed 均 &gt;0.5**。**auto 三 seed 在诊断上均 flipped=True**，**legacy 三 seed 均 flipped=False**（单 LCC 未达到翻转阈值/或相关定义不同），说明 **多探针+度 在本图上改变了是否翻转的决策，且对指标有利**。 |
| **disney** | **auto 明显拉低** mean AUC 与 mean AP；**auto 的 flipped 全 False** 与 **部分 legacy 的 flipped True** 并存。需结合**是否应翻**的语义，不宜仅凭「少翻」论优劣。 |
| **books** | **legacy 三 seed 均 AUC&gt;0.5 且全翻转**；**auto 三 seed 均 AUC&lt;0.5 且全不翻**。本批中 **单阈值 LCC 翻转在此数据上更利于 AUC**；**auto 因 1/1/1 票型未过 margin 而保守不翻**（见 4.1）。 |

### 5.2 与「AUC&lt;0.5 = 极性/方向类风险」的汇总

- **全四数据集、共 12 个 seed/策略组合**中：  
  - **legacy** 下 **AUC&lt;0.5** 出现 **2 次**（**disney·42、reddit·42**）。  
  - **auto** 下 **AUC&lt;0.5** 出现 **5 次**（**books 全 3** + **disney 两例：42 与 3407**；**不**把 disney·2026 的 0.5 计为 &lt;0.5）。  
- 因此，**以「&lt;0.5 次数是否下降」为唯一指标时，本批 auto 未整体优于 legacy**；但在 **reddit 上，auto 明显消除 &lt;0.5 且提升 mean AUC**。

### 5.3 三条一句话结论

1. **全四数据集并集**：`auto_vote` 未使 **AUC&lt;0.5 总次数**下降（本批 **2 → 5**），**未在整体意义上「解决」**所有图上的**分数方向/极性**问题。  
2. **reddit** 上 **auto_vote 有效**：三 seed 均 &gt;0.5、mean AUC 提升、且 **对本来 &lt;0.5 的 seed 有明确修复**。  
3. **books** 上 **本批显示 legacy 的「必翻」更合该评估目标**；**auto 偏保守不翻**与 **1/1/1 票型** 一致，属**调参（非改算法）**可探索方向（见 6 节）。

---

## 6. 后续工作建议（仅调参、不改代码逻辑）

1. 针对 **books**（与部分 **disney**）：在对应 `*_best.yaml` 中**小幅降低** `polarity_min_confidence` 或 **`polarity_vote_margin`**，使 **1/1/1** 类票型在 **sum_conf 足够**时有机会形成 **可翻** 决策。  
2. 微调 **polarity_lcc_rho_strong** / **polarity_deg_rho_strong**（**略降**可减 **abstain**、改变 flip/keep 比）。  
3. 在 **0.08–0.12** 内试 **polarity_vote_q**；配合 **polarity_connectivity_rel_gap** 小步扫。  
4. 固定 **3 seeds** 后重复 `run_polarity_eval.py` 出表，只比较 **同一文件** 下 `polarity_report_closure.md` 的 **ΔAUC 与 &lt;0.5 计数**。  
5. 始终用 **同一 conda 的 `--python`**，并保留 `polarity_eval_runs/*json` 以便断点续跑。

---

## 7. 相关文件索引

| 文件 | 内容 |
|------|------|
| `results/polarity_eval_runs/all_records.json` | 全 24 条结果（含 `polarity_diagnostics`） |
| `results/polarity_eval_runs/{dataset}_seed{seed}_{mode}.json` | 每次运行的完整 JSON（含 AUC、诊断） |
| `results/polarity_eval_summary.md` | 自动生成的对照、翻转率、探针行 |
| `results/polarity_report_closure.md` | 闭环表（ΔAUC、&lt;0.5 计数、seed 矩阵） |
| `results/polarity_full_run.log` | 全量标准输出日志（若曾用 `nohup` 重定向） |
| **本文** | `results/POLARITY_EVAL_FULL_REPORT.md` |

---

*报告生成自仓库内上述统计文件；若重新跑数，请同步更新本报告或重新导出。*
