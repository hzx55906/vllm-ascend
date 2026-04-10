# SPDX-License-Identifier: Apache-2.0
"""Rejection sampling functions for compressed logits in distributed setting."""

import torch
from vllm.v1.sample.metadata import SamplingMetadata
import triton
import triton.language as tl
import torch

def sample_recovered_tokens_compressed(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,  # [num_tokens, top_k*tp_size]
    target_indices: torch.Tensor,  # [num_tokens, top_k*tp_size] global vocab indices
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    """Sample recovered tokens from compressed probability distribution.

    When a draft token is rejected, we sample a "recovered" token from
    a modified distribution. This function handles the compressed case
    where target_probs and target_indices represent only the top-k candidates.

    Args:
        max_spec_len: Maximum speculative length
        num_draft_tokens: Number of draft tokens per request
        cu_num_draft_tokens: Cumulative draft tokens [batch_size]
        draft_token_ids: Draft token IDs [num_tokens]
        draft_probs: Draft probabilities [num_tokens, vocab_size] or None
        target_probs: Target probabilities (compressed) [num_tokens, top_k*tp_size]
        target_indices: Global vocabulary indices [num_tokens, top_k*tp_size]
        sampling_metadata: Sampling metadata
        device: Device

    Returns:
        recovered_token_ids: [num_tokens] recovered token IDs (global vocab indices)
    """
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    compressed_vocab_size = target_probs.shape[-1]

    # Create q distribution for sampling
    q = torch.empty(
        (batch_size, compressed_vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()

    num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
    has_draft_mask = num_draft_tensor > 0

    for i, generator in sampling_metadata.generators.items():
        if num_draft_tokens[i] > 0:
            temp_q = torch.empty_like(q[i])
            temp_q.exponential_(generator=generator)
            q[i] = torch.where(has_draft_mask[i], temp_q, q[i])

    recovered_token_ids = torch.empty_like(draft_token_ids)

    # Process each token
    cu_start = torch.cat([
        torch.tensor([0], pin_memory=True).to(device, non_blocking=True),
        cu_num_draft_tokens[:-1],
    ])
    cu_end = cu_num_draft_tokens

    token_indices = torch.arange(num_tokens, device=device)
    token_indices_expanded = token_indices[:, None]
    cu_start_expanded = cu_start[None, :]
    cu_end_expanded = cu_end[None, :]

    in_range_mask = (token_indices_expanded >= cu_start_expanded) & (token_indices_expanded < cu_end_expanded)
    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)
    has_match = in_range_mask.any(dim=1)
    token_to_batch = torch.where(has_match, token_to_batch, 0)

    # For compressed case, we need to compute the adjusted distribution
    # prob = max(0, target_probs - draft_probs_for_candidates)
    # where draft_probs_for_candidates are the draft probs at the candidate indices

    if draft_probs is None:
        # N-GRAM case: zero out the draft token in target_probs
        # Find where draft_token_id appears in target_indices
        # and set corresponding target_probs to 0
        prob = target_probs.clone()
        for i in range(num_tokens):
            draft_id = draft_token_ids[i]
            mask = target_indices[i] == draft_id
            prob[i, mask] = 0
    else:
        # Probabilistic case: max(0, target_probs - draft_probs_at_indices)
        # Gather draft probs at candidate indices
        # draft_probs: [num_tokens, vocab_size]
        # target_indices: [num_tokens, top_k*tp_size]
        # Need: draft_probs_at_indices: [num_tokens, top_k*tp_size]

        # Flatten indices for gather
        flat_indices = target_indices.flatten()  # [num_tokens * top_k*tp_size]
        token_offsets = torch.arange(num_tokens, device=device)[:, None] * draft_probs.shape[1]
        flat_token_offsets = token_offsets.expand_as(target_indices).flatten()

        # Gather draft probs at candidate positions
        draft_probs_flat = draft_probs.flatten()
        valid_mask = flat_indices < draft_probs.shape[1]
        flat_draft_probs_at_indices = torch.where(
            valid_mask,
            draft_probs_flat[flat_token_offsets + flat_indices],
            torch.tensor(0.0, device=device)
        )
        draft_probs_at_indices = flat_draft_probs_at_indices.view(num_tokens, compressed_vocab_size)

        # Compute adjusted probability
        prob = torch.maximum(
            target_probs - draft_probs_at_indices,
            torch.tensor(0.0, device=device),
        )

    # Sample using q distribution
    q_values = q[token_to_batch]  # [num_tokens, compressed_vocab_size]

    epsilon = 1e-10
    q_values_safe = torch.where(q_values == 0, epsilon, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), epsilon, q_values_safe)

    prob_over_q = prob / q_values_safe
    prob_over_q = torch.where((q_values == 0) | torch.isinf(q_values), -1e10, prob_over_q)

    # Get the index in compressed vocab
    compressed_indices = torch.argmax(prob_over_q, dim=1)

    # Convert to global vocabulary indices
    recovered_token_ids = target_indices[torch.arange(num_tokens, device=device), compressed_indices]

    return recovered_token_ids


def rejection_random_sample_compressed_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, top_k*tp_size]
    target_indices,  # [num_tokens, top_k*tp_size] global vocab indices
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,  # This is compressed_vocab_size = top_k*tp_size
    IS_NGRAM=False,
):
    """Rejection sampling for random sampling with compressed logits.

    This function implements speculative decoding rejection sampling when
    target probabilities are compressed to only top-k candidates.

    Key difference from uncompressed version:
    - draft_probs is full [num_tokens, vocab_size]
    - target_probs is compressed [num_tokens, top_k*tp_size]
    - target_indices maps compressed indices to global vocab indices
    - When draft_token is not in target_indices, it's automatically rejected
    """
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start

    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]

    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]  # [batch_size, max_draft_len]

    # For each draft token, check if it's in the target_indices
    # draft_tokens: [batch_size, max_draft_len]
    # target_indices: [num_tokens, top_k*tp_size]

    # Flatten for processing
    flat_global_indices = global_token_indices.flatten()  # [batch_size * max_draft_len]
    flat_draft_tokens = draft_tokens.flatten()  # [batch_size * max_draft_len]

    # Get target_indices for each token position
    flat_target_indices = target_indices[flat_global_indices]  # [batch_size * max_draft_len, top_k*tp_size]
    flat_target_probs = target_probs[flat_global_indices]  # [batch_size * max_draft_len, top_k*tp_size]

    # Check if draft token is in target candidates
    # Compare each draft token with its candidate indices
    draft_expanded = flat_draft_tokens.unsqueeze(1)  # [batch_size * max_draft_len, 1]
    is_in_candidates = (flat_target_indices == draft_expanded)  # [batch_size * max_draft_len, top_k*tp_size]

    # Get the probability of draft token from target (if present)
    # If not present, probability is 0
    target_token_probs_flat = torch.where(
        is_in_candidates,
        flat_target_probs,
        torch.tensor(0.0, device=device)
    ).sum(dim=1)  # [batch_size * max_draft_len]

    target_token_probs = target_token_probs_flat.view(batch_size, max_draft_len)

    # Get draft probabilities
    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        # Gather draft probs at draft token positions
        flat_draft_probs = draft_probs[flat_global_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    # Acceptance condition
    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)

    acceptance_condition = (draft_token_probs > zero_threshold) & (
        target_token_probs / draft_token_probs >= uniform_token_probs
    )

    first_rejection = (~acceptance_condition) & valid_mask

    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)

    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True),
        first_rejection.float().argmax(dim=1, keepdim=True),
        default_pos
    )
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask

    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)

    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask

    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len]),
    )

    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask, final_tokens, output_token_ids[:, :max_draft_len]
    )

    # Add bonus tokens
    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection

    bonus_positions = num_draft_per_batch

    seq_len = output_token_ids.shape[1]
    all_positions_cpu = torch.arange(seq_len, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]

    batch_bonus_positions = bonus_positions[:, None]

    max_spec_len_cpu = torch.tensor([max_spec_len], pin_memory=True)
    max_spec_len_device = max_spec_len_cpu.to(device, non_blocking=True)

    valid_bonus_pos = bonus_positions < (max_spec_len_device + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos

    bonus_pos_match = all_positions == batch_bonus_positions
    bonus_pos_mask = bonus_pos_match & final_bonus_mask[:, None]

    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)


def rejection_random_sample_block_verify_compressed_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, top_k*tp_size]
    target_indices,  # [num_tokens, top_k*tp_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
):
    """Block verify rejection sampling with compressed logits."""
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = (cu_end - cu_start)[:, None]
    pos_indices_cpu = torch.arange(max_spec_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]
    valid_mask = pos_indices < num_draft_per_batch
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]

    # Flatten for processing
    flat_global_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()

    flat_target_indices = target_indices[flat_global_indices]
    flat_target_probs = target_probs[flat_global_indices]

    draft_expanded = flat_draft_tokens.unsqueeze(1)
    is_in_candidates = (flat_target_indices == draft_expanded)

    target_token_probs_flat = torch.where(
        is_in_candidates,
        flat_target_probs,
        torch.tensor(0.0, device=device)
    ).sum(dim=1)

    target_token_probs = target_token_probs_flat.view(batch_size, max_spec_len)

    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_draft_probs = draft_probs[flat_global_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_spec_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    # Block verify logic
    pi = target_token_probs / draft_token_probs
    pi = pi.clamp(max=1.0)
    pi = torch.cumprod(pi, dim=-1)
    uniform_token_probs = torch.cumprod(uniform_token_probs, dim=-1)
    legal_mask = (draft_token_probs > 0) & (pi >= uniform_token_probs)
    legal_mask = legal_mask & valid_mask

    last_accept_pos = torch.where(
        legal_mask.any(dim=-1, keepdim=True),
        (max_spec_len - legal_mask.flip(dims=[-1]).float().argmax(dim=-1, keepdim=True) - 1),
        -1,
    )
    non_greedy_mask = (~is_greedy)[:, None]

    accept_mask = (pos_indices <= last_accept_pos) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(accept_mask, draft_tokens, output_token_ids[:, :max_spec_len])

    reject_mask = (pos_indices == last_accept_pos + 1) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(reject_mask, recovered_tokens, output_token_ids[:, :max_spec_len])

    bonus_mask = (last_accept_pos + 1 >= num_draft_per_batch) & non_greedy_mask
    all_positions_cpu = torch.arange(max_spec_len + 1, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]
    bonus_pos_match = all_positions == num_draft_per_batch
    bonus_mask = bonus_mask & bonus_pos_match
    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, max_spec_len + 1)
    output_token_ids[:] = torch.where(bonus_mask, bonus_values_expanded, output_token_ids)

@triton.jit
def _sample_recovered_kernel(
    # Pointers to tensors
    target_probs_ptr,
    target_indices_ptr,
    draft_probs_ptr,  # Can be None, handled by scalar arg
    draft_token_ids_ptr,
    q_ptr,
    cu_num_draft_ptr,
    recovered_ids_ptr,
    # Shapes
    num_tokens,
    vocab_size,       # compressed_vocab_size
    full_vocab_size,  # original vocab size (for draft_probs bounds check)
    # Strides
    stride_tp_batch,
    stride_tp_vocab,
    stride_ti_batch,
    stride_ti_vocab,
    stride_dp_batch,
    stride_dp_vocab,
    stride_q_batch,
    stride_q_vocab,
    # Scalars / Control flags
    HAS_DRAFT_PROBS: tl.constexpr,  # Constexpr for branch elimination
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for sampling recovered tokens in speculative decoding.
    Each program instance processes one token (row).
    """
    # 1. 确定当前处理的 Token 索引
    token_idx = tl.program_id(0)
    
    # 边界检查：如果当前 token 索引超出总数，直接返回
    if token_idx >= num_tokens:
        return

    # 2. 确定 Batch 索引 (请求索引)
    # PyTorch 逻辑: 根据 cu_num_draft_tokens 找到 token_idx 属于哪个 batch
    # Triton 中由于无法动态搜索变长列表，这里假设 batch_size 较小，使用循环查找
    # 或者假设传入了一个 token_to_batch 映射表 (更高效，推荐在 wrapper 中计算)
    # 这里为了演示纯 Triton 逻辑，使用简单的线性扫描 (假设 batch_size 很小，如 <= 128)
    
    # 注意：实际高性能实现中，建议在 Python wrapper 中计算 token_to_batch 并作为参数传入
    # 这里演示如何在 Kernel 内处理逻辑
    
    batch_idx = 0
    # 假设 cu_num_draft_ptr 指向的数据布局为累积值
    # 我们需要找到第一个 cu_num_draft > token_idx 的索引
    # 由于 Triton 无法动态遍历长数组，此处假设 batch_size 很小
    # 实际使用时建议 pre-compute token_to_batch
    
    # 为了简化，我们假设 wrapper 传入了正确的 batch_idx (通过额外参数或预处理)
    # 这里我们使用一个简化的假设：batch_idx 可以通过外部计算传入
    # 但为了保持函数签名与原逻辑一致，我们这里假设 batch_size <= 256 并做检查
    
    # --- 修正方案：使用 Wrapper 传入的 batch_offsets ---
    # (见下方 Python wrapper 部分，我们预先计算好 batch 索引)
    
    # 这里我们假设 wrapper 已经计算好了 batch_indices 并传入了 q_ptr 的偏移
    # 但为了完整性，我们这里模拟一下查找逻辑 (仅适用于小 Batch)
    
    # 3. 加载当前 Token 的 Draft ID
    draft_id = tl.load(draft_token_ids_ptr + token_idx)

    # 4. 初始化指针
    tp_ptr = target_probs_ptr + token_idx * stride_tp_batch
    ti_ptr = target_indices_ptr + token_idx * stride_ti_batch
    dp_ptr = draft_probs_ptr + token_idx * stride_dp_batch # if exists

    # 5. 计算 Batch Index (用于获取 q 值)
    # 这一步在 Triton 中比较昂贵，建议在 host 端计算好 batch_indices tensor
    # 这里假设我们有一个辅助指针或者直接线性扫描 cu_num_draft
    # 由于演示目的，我们假设 wrapper 传入了一个 `token_to_batch_ptr`
    
    # --- 假设我们在 wrapper 中计算了 token_to_batch ---
    # (下方 wrapper 代码中会生成这个 tensor)
    
    # 6. 循环处理 Vocab 维度的计算
    # 我们需要找到 max(prob / q)
    # prob = max(0, target_prob - draft_prob_at_idx)
    
    # 初始化最大值和对应的索引
    max_val = -1e10
    max_idx = 0
    
    # 获取当前 token 对应的 batch 索引 (假设已加载或计算)
    # 这里我们简化逻辑，假设 batch_idx 已知
    
    # 遍历 compressed vocab
    # 使用 BLOCK_SIZE 进行分块循环
    for vocab_offset in range(0, vocab_size, BLOCK_SIZE):
        vocab_offsets = vocab_offset + tl.arange(0, BLOCK_SIZE)
        mask = vocab_offsets < vocab_size
        
        # 加载 target_probs
        t_prob = tl.load(tp_ptr + vocab_offsets * stride_tp_vocab, mask=mask, other=0.0)
        # 加载 target_indices
        t_idx = tl.load(ti_ptr + vocab_offsets * stride_ti_vocab, mask=mask, other=0)
        
        # 计算 draft_prob_at_idx
        d_prob = 0.0
        if HAS_DRAFT_PROBS:
            # draft_probs 是 [num_tokens, full_vocab_size]
            # 我们需要根据 t_idx 从 draft_probs 的对应行取出概率
            # t_idx 是全局索引
            
            # 检查索引是否越界 (full_vocab_size)
            valid_idx = t_idx < full_vocab_size
            # 计算 draft_probs 的偏移
            # dp_ptr 指向当前 token 行的首地址
            d_prob_candidate = tl.load(dp_ptr + t_idx * stride_dp_vocab, mask=mask & valid_idx, other=0.0)
            d_prob = d_prob_candidate
        
        # 计算 adjusted prob
        # prob = max(0, target - draft)
        adj_prob = tl.maximum(0.0, t_prob - d_prob)
        
        # 特殊处理：如果是 N-Gram (无 draft_probs)，且 t_idx == draft_id，则 prob = 0
        # 这对应原代码的 if draft_probs is None 分支
        if not HAS_DRAFT_PROBS:
            # 如果候选索引等于当前 draft_token_id，概率置 0
            is_draft_token = t_idx == draft_id
            adj_prob = tl.where(is_draft_token, 0.0, adj_prob)

        # 加载 q 值
        # q 的形状是 [batch_size, vocab_size]
        # 我们需要知道当前 token 属于哪个 batch
        # 假设我们有 batch_idx (通过某种方式获取)
        
        # 这里为了 Kernel 完整性，我们使用一个技巧：
        # q_ptr 已经在 wrapper 中被处理成了 [num_tokens, vocab_size] 的形状
        # 或者我们传入 batch_indices tensor
        # 假设 wrapper 传入 q_row_ptr 指向当前 token 对应的 q 行
        
        # 7. 加载 q 并计算 ratio
        # 假设 wrapper 传入的 q_ptr 已经扩展对齐到 num_tokens
        q_val = tl.load(q_ptr + token_idx * stride_q_batch + vocab_offsets * stride_q_vocab, mask=mask, other=1.0)
        
        # 安全处理 q=0 或 inf
        q_safe = tl.where(q_val == 0.0, 1e-10, q_val)
        q_safe = tl.where(q_val > 1e9, 1e-10, q_safe) # handle inf approx
        
        ratio = adj_prob / q_safe
        
        # 处理无效 ratio
        ratio = tl.where((q_val == 0.0) | (q_val > 1e9), -1e10, ratio)
        
        # 8. 更新最大值
        # 我们需要跨 Block 维度进行比较
        # 这里使用 tl.max 和 tl.argmax 只能在当前 block 内部比较
        # 因此我们需要手动维护 max_val 和 max_idx
        
        # 找出当前 block 内的最大值
        block_max_val = tl.max(ratio, axis=0) # 这里的 axis=0 是指 reduce arange
        # 找出最大值的索引 (相对于 block start)
        block_max_idx_local = tl.argmax(ratio, axis=0)
        block_max_idx_global = vocab_offset + block_max_idx_local
        
        # 更新全局最大值
        # Triton 不支持跨 loop 的动态 if，我们需要用比较运算
        is_new_max = block_max_val > max_val
        max_val = tl.where(is_new_max, block_max_val, max_val)
        max_idx = tl.where(is_new_max, block_max_idx_global, max_idx)

    # 9. 写回结果
    # recovered_token_ids = target_indices[token_idx, max_idx]
    recovered_id = tl.load(ti_ptr + max_idx * stride_ti_vocab)
    tl.store(recovered_ids_ptr + token_idx, recovered_id)


def sample_recovered_tokens_compressed_triton(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,  # [num_tokens, top_k*tp_size]
    target_indices: torch.Tensor,  # [num_tokens, top_k*tp_size]
    sampling_meta: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    """
    Triton-wrapper for recovered token sampling.
    """
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    compressed_vocab_size = target_probs.shape[-1]
    full_vocab_size = draft_probs.shape[1] if draft_probs is not None else 0

    # 1. 准备 q 分布
    # 原逻辑中 q 是 [batch_size, vocab_size]，我们需要将其扩展对齐到 [num_tokens, vocab_size]
    # 以便 Kernel 能够直接按 token_idx 访问
    
    q_batch = torch.empty(
        (batch_size, compressed_vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q_batch.exponential_()

    # 处理 generators
    num_draft_tensor = torch.tensor(num_draft_tokens, device=device)
    has_draft_mask = num_draft_tensor > 0
    
    # 这部分保留 PyTorch 实现，因为涉及复杂的 Python 对象交互
    for i, generator in sampling_metadata.generators.items():
        if num_draft_tokens[i] > 0:
            temp_q = torch.empty_like(q_batch[i])
            temp_q.exponential_(generator=generator)
            q_batch[i] = torch.where(has_draft_mask[i], temp_q, q_batch[i])

    # 2. 构建 token_to_batch 映射
    # 将 q [B, V] 扩展为 q_expanded [T, V]，这样 Kernel 只需要按行索引
    # 这一步避免了 Kernel 内部复杂的 batch 查找逻辑
    
    # 计算 token_to_batch
    cu_start = torch.cat([
        torch.tensor([0], device=device),
        cu_num_draft_tokens[:-1],
    ])
    cu_end = cu_num_draft_tokens
    
    # 使用 searchsorted 或广播找到 batch_idx
    # broadcast: [T, 1] vs [1, B]
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(1)
    batch_indices_expanded = torch.arange(batch_size, device=device).unsqueeze(0)
    
    # 找到 token_indices < cu_end 的最小 batch index
    # 或者使用简单的 mask 逻辑
    # 这里使用 PyTorch 的 max 操作模拟原逻辑
    in_range_mask = (token_indices >= cu_start[batch_indices_expanded]) & \
                    (token_indices < cu_end[batch_indices_expanded])
    
    # argmax 得到 batch index (第一个 True 的位置)
    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)
    
    # 扩展 q: q_expanded[token_idx] = q_batch[batch_idx]
    q_expanded = q_batch[token_to_batch]  # [num_tokens, compressed_vocab_size]

    # 3. 准备输出
    recovered_token_ids = torch.empty(num_tokens, dtype=torch.long, device=device)

    # 4. Launch Kernel
    # 处理 grid 和 block size
    grid = (num_tokens,)
    BLOCK_SIZE = 128 # Triton 建议的 block size，可根据 vocab 调整
    
    # 处理 draft_probs 为 None 的情况
    HAS_DRAFT_PROBS = draft_probs is not None
    
    # 如果 draft_probs 为 None，我们需要传一个 dummy tensor 或者利用 constexpr 分支
    # Triton 要求指针参数必须有值，我们可以传一个空 tensor 的指针，Kernel 内部不会访问
    draft_probs_dummy = torch.empty(0, device=device) if not HAS_DRAFT_PROBS else draft_probs

    _sample_recovered_kernel[grid](
        target_probs,
        target_indices,
        draft_probs_dummy,
        draft_token_ids,
        q_expanded,  # 传入扩展后的 q
        cu_num_draft_tokens,
        recovered_token_ids,
        num_tokens,
        compressed_vocab_size,
        full_vocab_size,
        # Strides
        target_probs.stride(0),
        target_probs.stride(1),
        target_indices.stride(0),
        target_indices.stride(1),
        draft_probs_dummy.stride(0) if HAS_DRAFT_PROBS else 0,
        draft_probs_dummy.stride(1) if HAS_DRAFT_PROBS else 0,
        q_expanded.stride(0),
        q_expanded.stride(1),
        # Constexprs
        HAS_DRAFT_PROBS=HAS_DRAFT_PROBS,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return recovered_token_ids




