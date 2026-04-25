"""
工具函数
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def compute_node_lcc_tensor(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    无向图下每个节点的局部聚类系数 (NetworkX)，shape [N]，CPU float32。
    """
    import networkx as nx

    ei = edge_index.detach().cpu()
    data = Data(edge_index=ei, num_nodes=int(num_nodes))
    G = to_networkx(data, to_undirected=True)
    lcc_dict = nx.clustering(G)
    out = np.zeros(int(num_nodes), dtype=np.float32)
    for i in range(int(num_nodes)):
        out[i] = float(lcc_dict.get(i, 0.0))
    return torch.from_numpy(out)


def compute_node_degree_tensor(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    无向图下每个节点度数（行/列无向合一度），shape [N]，与 edge_index 同设备。
    """
    n = int(num_nodes)
    dev = edge_index.device
    one = torch.ones(edge_index.size(1), device=dev, dtype=torch.float32)
    deg = torch.zeros(n, device=dev, dtype=torch.float32)
    deg.index_add_(0, edge_index[0], one)
    deg.index_add_(0, edge_index[1], one)
    return deg


def _linear_flip01_numpy_style(score: torch.Tensor) -> torch.Tensor:
    """与 calibrate_polarity_lcc_spearman / flip_score 同口径的 [0,1] 线性翻转（张量，带梯度无需求）。"""
    with torch.no_grad():
        smin, smax = score.min(), score.max()
        if float(smax - smin) <= 1e-12:
            return -score
        return 1.0 - (score - smin) / (smax - smin)


def _spearman_rho(
    a: np.ndarray, b: np.ndarray
) -> Optional[float]:
    r, _ = spearmanr(a, b)
    if r is None or (isinstance(r, float) and (np.isnan(r))):
        return None
    return float(r)


def _undirected_pair_set(edge_index: np.ndarray, n: int) -> int:
    """去重后的无向边数（不重复计入双向）。"""
    seen = set()
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        seen.add((u, v))
    return len(seen)


def _induced_undirected_unique_in_top(
    edge_index: np.ndarray, top_set: set
) -> int:
    """诱导子图上的无向边数（对 u<v 去重，避免同一条边在双向存储时算两次）。"""
    seen = set()
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u not in top_set or v not in top_set:
            continue
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
    return len(seen)


def calibrate_polarity_auto_vote(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    lcc: torch.Tensor,
    degree: torch.Tensor,
    *,
    q: float = 0.1,
    margin: int = 1,
    min_confidence: float = 0.2,
    lcc_rho_strong: float = 0.04,
    deg_rho_strong: float = 0.04,
    connectivity_rel_gap: float = 0.02,
    legacy_lcc_threshold: float = -0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """
    多探针无监督极性「投票」：LCC-Spearman、度-Spearman、top-q 子图边密度与全图比。
    各探针输出 keep / flip / abstain，仅当 flip 票 >= keep 票 + margin 且总置信度达阈值时做线性翻转。
    返回: (calibrated_score, did_flip, diagnostics)
    """
    with torch.no_grad():
        diags: Dict[str, Any] = {
            "mode": "auto_vote",
            "probes": [],
        }
        dev = score.device
        score_np = score.detach().cpu().numpy().ravel()
        lcc_np = lcc.detach().cpu().numpy().ravel()
        deg_np = degree.detach().cpu().numpy().ravel()
        n = int(score_np.size)
        if n < 2 or lcc_np.size != n or deg_np.size != n:
            return score, False, {**diags, "error": "size_mismatch", "flipped": False}
        if np.std(score_np) <= 1e-12:
            if verbose:
                print("[auto_vote] skip: zero variance in score", flush=True)
            return score, False, {**diags, "abstain_reason": "zero_var", "flipped": False}
        q = float(np.clip(q, 0.01, 0.49))
        margin = int(max(0, margin))

        ei_np = edge_index.detach().cpu().numpy()
        if ei_np.size == 0:
            m_full = 0.0
            dens_full = 0.0
        else:
            m_uniq = _undirected_pair_set(ei_np, n)
            m_full = float(n * (n - 1) / 2.0) + 1e-12
            dens_full = float(2.0 * m_uniq) / (float(n * max(n - 1, 1)) + 1e-12)

        flip_v, keep_v = 0, 0
        conf_pieces: List[Tuple[str, float]] = []

        # —— Probe 1: rho(score, LCC) ——
        rho1 = _spearman_rho(score_np, lcc_np)
        v1: str
        c1 = 0.0
        if rho1 is None:
            v1 = "abstain"
        else:
            c1 = min(1.0, abs(rho1))
            if rho1 < float(legacy_lcc_threshold):
                v1 = "flip"
            elif rho1 > float(lcc_rho_strong):
                v1 = "keep"
            else:
                v1 = "abstain"
        if v1 == "flip":
            flip_v += 1
        elif v1 == "keep":
            keep_v += 1
        conf_pieces.append(("lcc", c1))
        diags["probes"].append(
            {
                "name": "lcc_spearman",
                "rho": None if rho1 is None else float(rho1),
                "vote": v1,
                "confidence": float(c1),
            }
        )

        # —— Probe 2: rho(score, degree) ——
        rho2 = _spearman_rho(score_np, deg_np)
        v2: str
        c2 = 0.0
        if rho2 is None or np.std(deg_np) <= 1e-12:
            v2 = "abstain"
        else:
            c2 = min(1.0, abs(rho2))
            if rho2 < float(legacy_lcc_threshold):
                v2 = "flip"
            elif rho2 > float(deg_rho_strong):
                v2 = "keep"
            else:
                v2 = "abstain"
        if v2 == "flip":
            flip_v += 1
        elif v2 == "keep":
            keep_v += 1
        conf_pieces.append(("deg", c2))
        diags["probes"].append(
            {
                "name": "deg_spearman",
                "rho": None if rho2 is None else float(rho2),
                "vote": v2,
                "confidence": float(c2),
            }
        )

        # —— Probe 3: top-q 诱导边密度 相对 全图 ——
        k = max(2, int(q * n))
        order = np.argsort(-score_np)
        top_idx = order[:k]
        top_set = set(int(t) for t in top_idx.tolist())
        n_pairs = k * (k - 1) // 2
        m_top = _induced_undirected_unique_in_top(ei_np, top_set)
        dens_top = float(m_top) / (float(n_pairs) + 1e-12)
        d_gap = float(dens_top) - float(dens_full)
        c3 = min(1.0, abs(d_gap) * 5.0)  # 0~1 量纲
        v3: str
        if dens_top < float(dens_full) - float(connectivity_rel_gap):
            v3 = "flip"
        elif dens_top > float(dens_full) + float(connectivity_rel_gap):
            v3 = "keep"
        else:
            v3 = "abstain"
        if v3 == "flip":
            flip_v += 1
        elif v3 == "keep":
            keep_v += 1
        conf_pieces.append(("topq_density", c3))
        diags["probes"].append(
            {
                "name": "topq_density",
                "dens_top": float(dens_top),
                "dens_full": float(dens_full),
                "gap": float(d_gap),
                "k_top": k,
                "m_induced": m_top,
                "vote": v3,
                "confidence": float(c3),
            }
        )

        total_conf = float(sum(p[1] for p in conf_pieces))
        diags["total_confidence"] = total_conf
        diags["sum_confidence"] = total_conf
        diags["flip_votes"] = flip_v
        diags["keep_votes"] = keep_v
        diags["margin"] = margin
        should_flip = (
            (flip_v >= keep_v + margin) and (total_conf >= float(min_confidence)) and (flip_v >= 1)
        )
        diags["flipped"] = bool(should_flip)
        if rho1 is not None:
            diags["rho_lcc"] = float(rho1)
        if rho2 is not None:
            diags["rho_deg"] = float(rho2)
        if verbose or should_flip:
            print(
                f"[auto_vote] rho_lcc={rho1} rho_deg={rho2} | flip/keep/margin={flip_v}/{keep_v}/{margin} | "
                f"conf={total_conf:.4f} (min {min_confidence}) -> {'FLIP' if should_flip else 'keep score'}",
                flush=True,
            )
        if should_flip:
            return _linear_flip01_numpy_style(score.to(dev)), True, diags
        return score, False, diags


def calibrate_polarity_lcc_spearman(
    score: torch.Tensor,
    lcc: torch.Tensor,
    threshold: float = -0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """
    基于 Score 与局部聚类系数 LCC 的 Spearman 秩相关，无监督极性探针。
    rho < threshold 时对 score 做 [0,1] 线性翻转（与 flip_score 一致）。
    返回: (score, flipped, diagnostics)
    """
    with torch.no_grad():
        diags: Dict[str, Any] = {"mode": "legacy_lcc", "flipped": False, "rho_lcc": None, "legacy_threshold": float(threshold)}
        score_np = score.detach().cpu().numpy().ravel()
        lcc_np = lcc.detach().cpu().numpy().ravel()
        n = score_np.size
        if n < 2 or lcc_np.size != n:
            return score, False, {**diags, "abstain_reason": "size_mismatch"}
        if np.std(score_np) <= 1e-12:
            if verbose:
                print("[LCC-Spearman] skip: zero variance in score", flush=True)
            return score, False, {**diags, "abstain_reason": "zero_var"}
        rho, _p = spearmanr(score_np, lcc_np)
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            if verbose:
                print("[LCC-Spearman] skip: Spearman undefined (e.g. constant LCC)", flush=True)
            return score, False, {**diags, "abstain_reason": "rho_nan"}
        rho_f = float(rho)
        diags["rho_lcc"] = rho_f
        if verbose:
            print(f"[LCC-Spearman] rho(score, LCC)={rho_f:.4f} (threshold={threshold})", flush=True)
        if rho_f < threshold:
            smin, smax = score.min(), score.max()
            if float(smax - smin) <= 1e-12:
                diags["flipped"] = True
                return -score, True, diags
            out = 1.0 - (score - smin) / (smax - smin)
            diags["flipped"] = True
            return out, True, diags
        return score, False, diags


def softmax_with_temperature(input, t=1, axis=-1):
    """
    带温度参数的 softmax 函数
    
    参数:
        input: 输入张量
        t: 温度参数，t > 1 时分布更平滑，t < 1 时分布更尖锐
        axis: 应用 softmax 的维度
    
    返回:
        归一化后的概率分布
    """
    return F.softmax(input / t, dim=axis)
