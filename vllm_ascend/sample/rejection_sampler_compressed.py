# SPDX-License-Identifier: Apache-2.0
"""Rejection sampling functions for compressed logits in distributed setting."""

import torch
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ops.triton.reject_sample import (
    sample_recovered_tokens_compressed_kernel,
    rejection_random_sample_compressed_kernel,
    rejection_random_sample_block_verify_compressed_kernel,
    cal_grid_and_block_size,
)


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

    # Use Triton kernel if available
    if HAS_TRITON:
        sample_recovered_tokens_compressed_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            target_indices,
            q,
            compressed_vocab_size,
            triton.next_power_of_2(compressed_vocab_size),
            NO_DRAFT_PROBS=draft_probs is None,
            SUB_BLOCK=512,
        )
        return recovered_token_ids

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

    # Use Triton kernel if available
    if HAS_TRITON:
        vec_len = batch_size
        grid, block_size = cal_grid_and_block_size(batch_size)
        rejection_random_sample_compressed_kernel[(grid,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            target_indices,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs.to(torch.float32),
            is_greedy,
            max_spec_len,
            vocab_size,
            # Note: vocab_size here is compressed_vocab_size
            # We need original vocab_size for draft_probs indexing
            # draft_probs.shape[-1] if draft_probs is not None else vocab_size,
            vec_len,
            NO_DRAFT_PROBS=draft_probs is None,
            BLOCK_SIZE=block_size,
        )
        return

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

    # Use Triton kernel if available
    if HAS_TRITON:
        vec_len = batch_size
        grid, block_size = cal_grid_and_block_size(batch_size)
        rejection_random_sample_block_verify_compressed_kernel[(grid,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            target_indices,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs.to(torch.float32),
            is_greedy,
            max_spec_len,
            draft_probs.shape[-1] if draft_probs is not None else vocab_size,
            vocab_size,
            vec_len,
            NO_DRAFT_PROBS=draft_probs is None,
            BLOCK_SIZE=block_size,
        )
        return

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