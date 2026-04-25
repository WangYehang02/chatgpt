# 无监督极性评估 — auto_vote vs legacy_lcc

Seeds: [42, 3407, 2026]。本表由 `run_polarity_eval.py` 在本地训练运行后自动写入。

## 新旧策略对照表（每数据集 mean±std AUC / AP）

| dataset | auto_vote AUC | legacy_lcc AUC | auto_vote AP | legacy_lcc AP |
|---------|--------------|----------------|--------------|---------------|
| books | 0.4271±0.0134 | 0.6071±0.0173 | 0.0188±0.0022 | 0.0292±0.0011 |
| disney | 0.3964±0.0939 | 0.6671±0.1422 | 0.0444±0.0054 | 0.1018±0.0326 |
| enron | 0.8111±0.0449 | 0.8223±0.0514 | 0.0104±0.0095 | 0.0111±0.0108 |
| reddit | 0.5626±0.0232 | 0.5082±0.0267 | 0.0379±0.0031 | 0.0313±0.0017 |

## 极性翻转触发率（有诊断的 run）

| dataset | auto_vote | legacy_lcc |
|---------|----------|------------|
| books | 0.0% (n=3) | 100.0% (n=3) |
| disney | 0.0% (n=3) | 66.7% (n=3) |
| enron | 100.0% (n=3) | 100.0% (n=3) |
| reddit | 100.0% (n=3) | 0.0% (n=3) |

## 逐次运行明细（rho、votes、探针票型）

- **books** seed=42 mode=**auto_vote**  AUC=0.4385 AP=0.0213  flipped=False  rho_lcc=-0.052829337879332215 rho_deg=0.558280878481317
    - fv/kv/m=1/1/1 sum_conf=0.6631361825995169  probes=[lcc_spearman:flip, deg_spearman:keep, topq_density:abstain]
- **books** seed=2026 mode=**auto_vote**  AUC=0.4083 AP=0.0159  flipped=False  rho_lcc=-0.1500713201153132 rho_deg=0.6883527663464467
    - fv/kv/m=1/1/1 sum_conf=0.9355361722548322  probes=[lcc_spearman:flip, deg_spearman:keep, topq_density:abstain]
- **books** seed=3407 mode=**auto_vote**  AUC=0.4345 AP=0.0192  flipped=False  rho_lcc=-0.07288309392784612 rho_deg=0.5429257972319793
    - fv/kv/m=1/1/1 sum_conf=0.6490911897188552  probes=[lcc_spearman:flip, deg_spearman:keep, topq_density:abstain]
- **books** seed=42 mode=**legacy_lcc**  AUC=0.6216 AP=0.0292  flipped=True  rho_lcc=-0.06462062037518355 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **books** seed=2026 mode=**legacy_lcc**  AUC=0.6170 AP=0.0306  flipped=True  rho_lcc=-0.15006777712625002 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **books** seed=3407 mode=**legacy_lcc**  AUC=0.5828 AP=0.0279  flipped=True  rho_lcc=-0.08591094951366433 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **disney** seed=42 mode=**auto_vote**  AUC=0.2726 AP=0.0368  flipped=False  rho_lcc=-0.1311922286266675 rho_deg=0.7338606032172544
    - fv/kv/m=1/2/1 sum_conf=1.7817731427420285  probes=[lcc_spearman:flip, deg_spearman:keep, topq_density:keep]
- **disney** seed=2026 mode=**auto_vote**  AUC=0.5000 AP=0.0484  flipped=False  rho_lcc=None rho_deg=None
    - fv/kv/m=0/1/1 sum_conf=0.6136900078678081  probes=[lcc_spearman:abstain, deg_spearman:abstain, topq_density:keep]
- **disney** seed=3407 mode=**auto_vote**  AUC=0.4167 AP=0.0481  flipped=False  rho_lcc=0.045838172307908945 rho_deg=0.2806825936296645
    - fv/kv/m=1/1/1 sum_conf=0.4704065156455115  probes=[lcc_spearman:abstain, deg_spearman:keep, topq_density:flip]
- **disney** seed=42 mode=**legacy_lcc**  AUC=0.4661 AP=0.0558  flipped=False  rho_lcc=0.0634443845625828 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **disney** seed=2026 mode=**legacy_lcc**  AUC=0.7726 AP=0.1257  flipped=True  rho_lcc=-0.12773996810009502 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **disney** seed=3407 mode=**legacy_lcc**  AUC=0.7627 AP=0.1240  flipped=True  rho_lcc=-0.11379802503861174 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **enron** seed=42 mode=**auto_vote**  AUC=0.8526 AP=0.0060  flipped=True  rho_lcc=-0.06854491605573847 rho_deg=-0.13181948369589072
    - fv/kv/m=2/0/1 sum_conf=0.22554453580114148  probes=[lcc_spearman:flip, deg_spearman:flip, topq_density:abstain]
- **enron** seed=2026 mode=**auto_vote**  AUC=0.7487 AP=0.0235  flipped=True  rho_lcc=-0.10289223275312893 rho_deg=0.02768992422870895
    - fv/kv/m=1/0/1 sum_conf=0.18053192614995145  probes=[lcc_spearman:flip, deg_spearman:abstain, topq_density:abstain]
- **enron** seed=3407 mode=**auto_vote**  AUC=0.8321 AP=0.0016  flipped=True  rho_lcc=-0.07165881679071318 rho_deg=-0.11974500905482058
    - fv/kv/m=2/0/1 sum_conf=0.21907131194337173  probes=[lcc_spearman:flip, deg_spearman:flip, topq_density:abstain]
- **enron** seed=42 mode=**legacy_lcc**  AUC=0.8515 AP=0.0046  flipped=True  rho_lcc=-0.0643938340786459 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **enron** seed=2026 mode=**legacy_lcc**  AUC=0.7500 AP=0.0263  flipped=True  rho_lcc=-0.10222921091412966 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **enron** seed=3407 mode=**legacy_lcc**  AUC=0.8653 AP=0.0024  flipped=True  rho_lcc=-0.08481007395573802 rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **reddit** seed=42 mode=**auto_vote**  AUC=0.5930 AP=0.0423  flipped=True  rho_lcc=None rho_deg=-0.4864143587326197
    - fv/kv/m=1/0/1 sum_conf=0.48907980782810495  probes=[lcc_spearman:abstain, deg_spearman:flip, topq_density:abstain]
- **reddit** seed=2026 mode=**auto_vote**  AUC=0.5368 AP=0.0351  flipped=True  rho_lcc=None rho_deg=-0.5654760796762174
    - fv/kv/m=1/0/1 sum_conf=0.5679505791209745  probes=[lcc_spearman:abstain, deg_spearman:flip, topq_density:abstain]
- **reddit** seed=3407 mode=**auto_vote**  AUC=0.5579 AP=0.0364  flipped=True  rho_lcc=None rho_deg=-0.6006097341684409
    - fv/kv/m=1/0/1 sum_conf=0.6032253703115623  probes=[lcc_spearman:abstain, deg_spearman:flip, topq_density:abstain]
- **reddit** seed=42 mode=**legacy_lcc**  AUC=0.4754 AP=0.0293  flipped=False  rho_lcc=None rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **reddit** seed=2026 mode=**legacy_lcc**  AUC=0.5409 AP=0.0334  flipped=False  rho_lcc=None rho_deg=None
    - fv/kv/m=None/None/ sum_conf=
- **reddit** seed=3407 mode=**legacy_lcc**  AUC=0.5084 AP=0.0311  flipped=False  rho_lcc=None rho_deg=None
    - fv/kv/m=None/None/ sum_conf=

---
*说明：legacy 行对应 `polarity_mode: legacy_lcc` 与 LCC-Spearman 单阈值；auto 行对应当前 `*_best.yaml` 的 `auto_vote`。*
