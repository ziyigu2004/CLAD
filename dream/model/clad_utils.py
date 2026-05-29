import torch

def _symmetrize_attention(attn: torch.Tensor, mode: str = "max") -> torch.Tensor:
    rel = attn.to(torch.float32)
    if mode == "max":
        rel = torch.maximum(rel, rel.transpose(-1, -2))
    elif mode == "mean":
        rel = 0.5 * (rel + rel.transpose(-1, -2))
    else:
        raise ValueError(f"Unsupported sym_mode: {mode}")
    L = rel.size(-1)
    eye = torch.eye(L, device=rel.device, dtype=torch.bool)
    if rel.dim() == 3:
        eye = eye.unsqueeze(0)
    elif rel.dim() != 2:
        raise ValueError(f"Expected attention shape [L,L] or [B,L,L], got {tuple(rel.shape)}")
    rel = rel.masked_fill(eye, 0.0)
    return rel

def _build_contiguous_clusters_tensor(cand_idx: torch.Tensor):
    if cand_idx.numel() == 0:
        empty = cand_idx.new_empty((0,), dtype=torch.long)
        return empty, empty
    cut = torch.nonzero(cand_idx[1:] != cand_idx[:-1] + 1, as_tuple=False).squeeze(-1)
    start_ptr = torch.cat([cand_idx.new_zeros(1), cut + 1])
    end_ptr = torch.cat([cut, cand_idx.new_full((1,), cand_idx.numel() - 1)])
    starts = cand_idx[start_ptr]
    ends = cand_idx[end_ptr]
    return starts, ends

def _build_cluster_conflict_graph_mutual_top1_gpu(
    rel_2d: torch.Tensor,
    cand_idx: torch.Tensor,
    conflict_abs_tau: float = 1.5,
):
    device = rel_2d.device
    dtype = rel_2d.dtype
    starts, ends = _build_contiguous_clusters_tensor(cand_idx)
    K = int(starts.numel())
    if K == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.empty((0,), dtype=torch.long, device=device)
        return empty_bool, starts, ends, empty_long
    if K == 1:
        conflict = torch.zeros((1, 1), dtype=torch.bool, device=device)
        weights = (ends - starts + 1).to(torch.long)
        return conflict, starts, ends, weights
    L = rel_2d.size(0)
    pos = torch.arange(L, device=device).unsqueeze(0)
    membership = (pos >= starts.unsqueeze(1)) & (pos <= ends.unsqueeze(1))
    weights = membership.sum(dim=1).to(torch.long)
    M = membership.to(dtype)
    to_cluster = rel_2d @ M.t() / weights.clamp_min(1).to(dtype).unsqueeze(0)
    neg_inf = torch.finfo(dtype).min
    score = to_cluster.unsqueeze(0).masked_fill(~membership.unsqueeze(-1), neg_inf).max(dim=1).values
    score = torch.maximum(score, score.transpose(0, 1))
    score.fill_diagonal_(0.0)
    row_mean = score.sum(dim=1) / max(K - 1, 1)
    eye = torch.eye(K, device=device, dtype=torch.bool)
    masked_score = score.masked_fill(eye, neg_inf)
    top1 = masked_score.argmax(dim=1)
    a = torch.arange(K, device=device)
    mutual = (top1[top1] == a)
    wab = score[a, top1]
    tau_abs = conflict_abs_tau / float(L)
    if K <= 2:
        keep = mutual & (wab >= tau_abs)
    else:
        keep = mutual & (wab >= row_mean) & (wab >= row_mean[top1])
    conflict = torch.zeros((K, K), dtype=torch.bool, device=device)
    aa = a[keep]
    bb = top1[keep]
    conflict[aa, bb] = True
    conflict[bb, aa] = True
    conflict.fill_diagonal_(False)
    return conflict, starts, ends, weights

def _maximum_weight_independent_set_matching_gpu(
    conflict: torch.Tensor,
    weights: torch.Tensor,
    tie_score: torch.Tensor = None,
):
    device = conflict.device
    K = int(conflict.size(0))
    if K == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    deg = conflict.sum(dim=1)
    isolated = torch.nonzero(deg == 0, as_tuple=False).squeeze(-1)
    edges = torch.nonzero(torch.triu(conflict, diagonal=1), as_tuple=False)
    if edges.numel() == 0:
        return isolated
    i = edges[:, 0]
    j = edges[:, 1]
    score = weights.float()
    if tie_score is not None:
        score = score + 1e-4 * tie_score.float()
    choose_i = score[i] >= score[j]
    chosen = torch.where(choose_i, i, j)
    selected = torch.cat([isolated, chosen], dim=0)
    return selected.unique()

def filter_transfer_index_with_clad(
    transfer_index: torch.Tensor,
    attn_mean: torch.Tensor,
    sym_mode: str = "max",
) -> torch.Tensor:
    # 注意：这个实现基于 GPU 版本，使用上面定义的辅助函数
    rel = _symmetrize_attention(attn_mean, mode=sym_mode)
    out = torch.zeros_like(transfer_index, dtype=torch.bool)
    B, L = transfer_index.shape
    pos = torch.arange(L, device=transfer_index.device).unsqueeze(0)
    for b in range(B):
        cand_idx = torch.nonzero(transfer_index[b], as_tuple=False).squeeze(-1)
        if cand_idx.numel() == 0:
            continue
        conflict, starts, ends, weights = _build_cluster_conflict_graph_mutual_top1_gpu(
            rel_2d=rel[b],
            cand_idx=cand_idx,
        )
        K = int(weights.numel())
        if K == 1:
            out[b, starts[0]:ends[0] + 1] = True
            continue
        sel = _maximum_weight_independent_set_matching_gpu(conflict, weights)
        if sel.numel() == 0:
            sel = torch.argmax(weights).view(1)
        sel_starts = starts[sel].unsqueeze(1)
        sel_ends = ends[sel].unsqueeze(1)
        sel_mask = ((pos >= sel_starts) & (pos <= sel_ends)).any(dim=0)
        out[b, sel_mask] = True
    return out
