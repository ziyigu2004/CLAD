import torch

from generate import get_transfer_index


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

    start_ptr = torch.cat([
        cand_idx.new_zeros(1),
        cut + 1,
    ])
    end_ptr = torch.cat([
        cut,
        cand_idx.new_full((1,), cand_idx.numel() - 1),
    ])

    starts = cand_idx[start_ptr]
    ends = cand_idx[end_ptr]
    return starts, ends


def _build_cluster_conflict_graph_mutual_top1_gpu(
    rel_2d: torch.Tensor,
    cand_idx: torch.Tensor,
    conflict_abs_tau: float = 1.2,
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
    pos = torch.arange(L, device=device).unsqueeze(0)         # [1, L]
    membership = (pos >= starts.unsqueeze(1)) & (pos <= ends.unsqueeze(1))  # [K, L]

    weights = membership.sum(dim=1).to(torch.long)            # [K]
    M = membership.to(dtype)                                  # [K, L]

    to_cluster = rel_2d @ M.t() / weights.clamp_min(1).to(dtype).unsqueeze(0)   # [L, K]

    neg_inf = torch.finfo(dtype).min

    score = to_cluster.unsqueeze(0).masked_fill(
        ~membership.unsqueeze(-1), neg_inf
    ).max(dim=1).values                                       # [K, K]

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
        keep = (
            mutual
            # & (wab >= tau_abs)
            & (wab >=  row_mean)
            & (wab >=  row_mean[top1])
        )

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
    """
    conflict 是 mutual-top1 生成的 matching graph。
    因此 MWIS 可以 exact vectorized 求解。
    """
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
):

    rel = _symmetrize_attention(attn_mean, mode=sym_mode)
    out = torch.zeros_like(transfer_index, dtype=torch.bool)

    B, L = transfer_index.shape
    pos = torch.arange(L, device=transfer_index.device).unsqueeze(0)  # [1, L]

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

        sel_starts = starts[sel].unsqueeze(1)   # [M, 1]
        sel_ends = ends[sel].unsqueeze(1)       # [M, 1]

        sel_mask = ((pos >= sel_starts) & (pos <= sel_ends)).any(dim=0)
        out[b, sel_mask] = True

    return out


@torch.no_grad()
def generate_clad(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    sym_mode="max",
):
    B = int(prompt.shape[0])
    Lp = int(prompt.shape[1])

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "steps must be divisible by number of blocks"
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt.to(model.device).clone()

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        for i in range(steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            out, attn_full = model(x, return_attn_scores=True)
            nfe += 1

            logits_blk = out.logits[:, s:e, :]
            attn_blk = attn_full[:, s:e, s:e]

            mask_blk = (x[:, s:e] == mask_id)

            x0_blk, transfer_idx_blk = get_transfer_index(logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, threshold)

            transfer_idx_blk = filter_transfer_index_with_clad(
                transfer_index=transfer_idx_blk,
                attn_mean=attn_blk,
                sym_mode=sym_mode,
            )

            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

    return x, nfe
