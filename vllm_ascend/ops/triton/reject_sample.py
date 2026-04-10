#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element, get_vectorcore_num


def cal_grid_and_block_size(batch_size: int):
    vectorcore_num = get_vectorcore_num()
    if batch_size <= vectorcore_num:
        grid = batch_size
        block_size = 1
    else:
        grid = vectorcore_num
        block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size


@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew_1(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(output_token_ids_ptr + position * 2 + 1, bonus_token_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_spec_len_1_triton(
    output_token_ids_ptr,  # [batch_size, 2]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,
    vec_len,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    draft_token_id = tl.load(draft_token_ids_ptr + offset, mask)
    target_argmax_id = tl.load(target_argmax_ptr + offset, mask)
    tl.store(output_token_ids_ptr + offset * 2, target_argmax_id, mask)

    # Add validity check for pos within the loop
    for pos in tl.range(0, BLOCK_SIZE):
        # Calculate the global position of the current token
        global_pos = block_idx * BLOCK_SIZE + pos
        if global_pos < vec_len:
            draft_token_id1 = get_element(draft_token_id, (pos,))
            target_argmax1 = get_element(target_argmax_id, (pos,))
            if draft_token_id1 == target_argmax1:
                bonus_renew_1(
                    bonus_token_ids_ptr,
                    global_pos,
                    output_token_ids_ptr,
                )


@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
    max_spec_len,
    num_tokens1,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(output_token_ids_ptr + position * (max_spec_len + 1) + num_tokens1, bonus_token_id)


@triton.jit(do_not_specialize=["vec_len", "max_spec_len"])
def rejection_greedy_sample_triton(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    vec_len,
    max_spec_len,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    if is_greedy_ptr is None:
        is_greedy_mask = mask
    else:
        is_greedy = tl.load(is_greedy_ptr + offset, mask=mask, other=0)
        is_greedy_mask = mask & (is_greedy != 0)

    start_idx = tl.where(offset == 0, 0, tl.load(cu_num_draft_tokens_ptr + offset - 1, is_greedy_mask))
    end_idx = tl.load(cu_num_draft_tokens_ptr + offset, is_greedy_mask)
    num_draft_tokens = end_idx - start_idx

    for pos in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_draft_tokens, (pos,))
        rejected = False
        start_idx1 = get_element(start_idx, (pos,))
        is_greedy_mask1 = get_element(is_greedy_mask, (pos,))
        position = block_idx * BLOCK_SIZE + pos
        for i in range(num_tokens1):
            if not rejected:
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx1 + i)
                target_argmax_id = tl.load(target_argmax_ptr + start_idx1 + i)
                tl.store(
                    output_token_ids_ptr + position * (max_spec_len + 1) + i,
                    target_argmax_id,
                )
                if draft_token_id != target_argmax_id:
                    # Reject.
                    rejected = True

        if not rejected and is_greedy_mask1:
            bonus_renew(
                bonus_token_ids_ptr,
                position,
                output_token_ids_ptr,
                max_spec_len,
                num_tokens1,
            )


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    vec_len,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(offsets == 0, 0, tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs
    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            start_idx = get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))
            for pos in range(num_draft_tokens):
                if not rejected:
                    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    if NO_DRAFT_PROBS:
                        draft_prob = 1
                    else:
                        draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
                    target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
                    uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                    # NOTE(woosuk): While the draft probability should never be 0,
                    # we check it to avoid NaNs. If it happens to be 0, we reject.
                    if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                        # Accept.
                        token_id = draft_token_id
                    else:
                        # Reject. Use recovered token.
                        rejected = True
                        token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
            if not rejected:
                # If all tokens are accepted, append the bonus token.
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                    bonus_token_id,
                )


@triton.jit(do_not_specialize=["replace_from", "replace_to", "vec_len"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    vec_len,
    MAX_NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offset = req_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vec_len

    start_idx = tl.where(offset == 0, 0, tl.load(cu_num_tokens_ptr + offset - 1, len_mask))
    end_idx = tl.load(cu_num_tokens_ptr + offset, len_mask)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + offset, len_mask)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)

    for i in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_tokens, (i,))
        start_idx1 = get_element(start_idx, (i,))
        src_val1 = get_element(src_val, (i,))
        offset1 = tl.arange(0, MAX_NUM_TOKENS)
        tl.store(output_ptr + start_idx1 + offset1, src_val1, mask=offset1 < num_tokens1)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
    global_recovered_id = -1
    global_max_p = -1.0
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        orig_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
        # Temporarily zero out the probability of the draft token.
        # This is essentially the same as target_prob - draft_prob, except that
        # n-gram does not have draft_prob. We regard it as 1.
        tl.store(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id, 0)
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                mask=vocab_offset < vocab_size,
                other=0,
            )
            q = tl.load(
                q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size, other=float("-inf")
            )
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = get_element(new_p, (recovered_id,))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id
    else:
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            draft_prob = tl.load(
                draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset, mask=vocab_offset < vocab_size, other=0
            )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                mask=vocab_offset < vocab_size,
                other=0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

            q = tl.load(
                q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size, other=float("-inf")
            )
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = get_element(new_p, (recovered_id,))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id

    tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)

    if NO_DRAFT_PROBS:
        # Restore the original probability.
        tl.store(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id, orig_prob)


def rejection_greedy_sample_with_triton(
    output_token_ids,
    num_draft_tokens,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    grid,
    block_size,
):
    vec_len = output_token_ids.shape[0]

    if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1 and is_greedy is None:
        rejection_greedy_sample_spec_len_1_triton[(grid,)](
            output_token_ids,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            vec_len,
            BLOCK_SIZE=block_size,
        )
    else:
        rejection_greedy_sample_triton[(grid,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            vec_len,
            max_spec_len,
            BLOCK_SIZE=block_size,
        )


def expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from, replace_to, max_num_tokens):
    vec_len = batch_size
    grid, block_size = cal_grid_and_block_size(batch_size)

    expand_kernel[(grid,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        vec_len,
        MAX_NUM_TOKENS=max_num_tokens,  # To avoid recompilation.
        BLOCK_SIZE=block_size,
    )


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_block_verify_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    vec_len,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(offsets == 0, 0, tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs
    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            pi = 1.0
            uniform_prob = 1.0
            last_accepted_token_pos = -1
            start_idx = get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))

            for pos in range(num_draft_tokens):
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
                tmp_uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                uniform_prob = uniform_prob * tmp_uniform_prob

                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)

                pi = min(pi * target_prob / draft_prob, 1.0)
                if draft_prob > 0 and pi >= uniform_prob:
                    last_accepted_token_pos = pos
                    rejected = False
                else:
                    rejected = True

            if last_accepted_token_pos > -1:
                for pos in range(last_accepted_token_pos + 1):
                    token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

            if rejected:
                recovered_token_id = tl.load(recovered_token_ids_ptr + start_idx + last_accepted_token_pos + 1)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1,
                    recovered_token_id,
                )
            else:
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)

# ============================================================================
# Compressed Logits Kernels for Distributed Sampling
# ============================================================================

@triton.jit
def sample_recovered_tokens_compressed_kernel(
    output_token_ids_ptr,       # [num_tokens] (int64 recommended)
    cu_num_draft_tokens_ptr,    # [batch_size]
    draft_token_ids_ptr,        # [num_tokens] global token ids
    draft_probs_ptr,            # [num_tokens, vocab_size] or None
    target_probs_ptr,           # [num_tokens, C]
    target_indices_ptr,         # [num_tokens, C] global vocab ids
    q_ptr,                      # [batch_size, C]
    compressed_vocab_size,      # C
    vocab_size,                 # global vocab size (for draft_probs gather bound check)
    NO_DRAFT_PROBS: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)

    start_idx = tl.where(req_idx == 0, 0, tl.load(cu_num_draft_tokens_ptr + req_idx - 1))
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos
    C = compressed_vocab_size

    n_loop = (C + SUB_BLOCK - 1) // SUB_BLOCK

    global_max_p = tl.full((), -1.0e20, tl.float32)
    global_recovered_id = tl.full((), -1, tl.int64)

    draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int64)

    for li in range(n_loop):
        c_start = li * SUB_BLOCK
        offs = c_start + tl.arange(0, SUB_BLOCK)
        mask = offs < C

        # target prob [SUB_BLOCK]
        tprob = tl.load(
            target_probs_ptr + token_idx * C + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # global vocab idx of candidates [SUB_BLOCK]
        gidx = tl.load(
            target_indices_ptr + token_idx * C + offs,
            mask=mask,
            other=0,
        ).to(tl.int64)

        if NO_DRAFT_PROBS:
            # ngram: zero probability at draft token id
            is_draft = (gidx == draft_token_id) & mask
            prob = tl.where(is_draft, 0.0, tprob)
        else:
            # probabilistic: max(0, target_prob - draft_prob_at_global_idx)
            valid = (gidx >= 0) & (gidx < vocab_size) & mask
            dprob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + gidx,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            prob = tl.maximum(tprob - dprob, 0.0)

        qv = tl.load(
            q_ptr + req_idx * C + offs,
            mask=mask,
            other=1.0,  # avoid div-by-zero on masked lanes
        ).to(tl.float32)

        bad_q = (qv <= 0) | tl.math.isinf(qv)
        score = tl.where(bad_q, -1.0e20, prob / qv)
        score = tl.where(mask, score, -1.0e20)

        block_best_score = tl.max(score, axis=0)
        block_best_idx = tl.argmax(score, axis=0).to(tl.int64)  # in [0, SUB_BLOCK)

        block_best_global_id = tl.load(
            target_indices_ptr + token_idx * C + (c_start + block_best_idx)
        ).to(tl.int64)

        better = block_best_score > global_max_p
        global_max_p = tl.where(better, block_best_score, global_max_p)
        global_recovered_id = tl.where(better, block_best_global_id, global_recovered_id)

    tl.store(output_token_ids_ptr + token_idx, global_recovered_id)

@triton.jit(do_not_specialize=["max_spec_len", "compressed_vocab_size"])
def rejection_random_sample_block_verify_compressed_kernel(
    output_token_ids_ptr,          # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,       # [batch_size]
    draft_token_ids_ptr,           # [num_tokens]
    draft_probs_ptr,               # [num_tokens, vocab_size] or None
    target_probs_ptr,              # [num_tokens, compressed_vocab_size]
    target_indices_ptr,            # [num_tokens, compressed_vocab_size] global vocab indices
    bonus_token_ids_ptr,           # [batch_size]
    recovered_token_ids_ptr,       # [num_tokens]
    uniform_probs_ptr,             # [num_tokens]
    is_greedy_ptr,                 # [batch_size]
    max_spec_len,
    vocab_size,                    # global vocab size (for draft_probs bounds check)
    compressed_vocab_size,         # C (local vocab size)
    vec_len,                       # batch_size
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    FIND_BLOCK: tl.constexpr = 32

    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    
    is_greedy = tl.load(is_greedy_ptr + offsets, mask=mask, other=1)
    not_greedy_mask = mask & (is_greedy == 0)
    
    start_idxs = tl.where(
        offsets == 0, 
        0, 
        tl.load(cu_num_draft_tokens_ptr + offsets - 1, mask=not_greedy_mask, other=0)
    )
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, mask=not_greedy_mask, other=0)
    n_num_draft_tokens = end_idxs - start_idxs

    for req_i in range(BLOCK_SIZE):
        not_greedy = tl.load(not_greedy_mask + req_i)
        
        if not_greedy:
            rejected = False
            pi = 1.0
            uniform_prob = 1.0
            last_accepted_token_pos = -1
            
            start_idx = tl.load(start_idxs + req_i)
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = tl.load(n_num_draft_tokens + req_i)

            for pos in range(num_draft_tokens):
                token_row = start_idx + pos
                draft_token_id = tl.load(draft_token_ids_ptr + token_row).to(tl.int64)
                
                target_prob = 0.0
                loops = (compressed_vocab_size + FIND_BLOCK - 1) // FIND_BLOCK
                
                for li in range(loops):
                    c_start = li * FIND_BLOCK
                    c_offs = c_start + tl.arange(0, FIND_BLOCK)
                    c_mask = c_offs < compressed_vocab_size

                    cand_ids = tl.load(
                        target_indices_ptr + token_row * compressed_vocab_size + c_offs,
                        mask=c_mask,
                        other=-1
                    ).to(tl.int64)
                    
                    hit = (cand_ids == draft_token_id) & c_mask
                    
                    cand_ps = tl.load(
                        target_probs_ptr + token_row * compressed_vocab_size + c_offs,
                        mask=hit,
                        other=0.0
                    )
                    
                    block_max = tl.max(cand_ps, axis=0)
                    target_prob = tl.maximum(target_prob, block_max)

                if NO_DRAFT_PROBS:
                    draft_prob = 1.0
                else:
                    in_range = (draft_token_id >= 0) & (draft_token_id < vocab_size)
                    draft_prob = tl.load(
                        draft_probs_ptr + token_row * vocab_size + draft_token_id,
                        mask=in_range,
                        other=0.0
                    )

                tmp_uniform_prob = tl.load(uniform_probs_ptr + token_row)
                uniform_prob = uniform_prob * tmp_uniform_prob
                
                pi = tl.minimum(pi * target_prob / draft_prob, 1.0)
                
                if draft_prob > 0 and pi >= uniform_prob:
                    last_accepted_token_pos = pos
                    rejected = False
                else:
                    rejected = True

            if last_accepted_token_pos > -1:
                for pos in range(last_accepted_token_pos + 1):
                    token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

            if rejected:
                recovered_token_id = tl.load(recovered_token_ids_ptr + start_idx + last_accepted_token_pos + 1)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1,
                    recovered_token_id,
                )
            else:
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)

@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_compressed_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, compressed_vocab_size]
    target_indices_ptr,  # [num_tokens, compressed_vocab_size] global vocab indices
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    compressed_vocab_size,
    vec_len, # batch_size
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    
    start_idxs = tl.where(offsets == 0, 0, tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs

    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            start_idx = get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))
            
            for pos in range(num_draft_tokens):
                if not rejected:
                    token_idx = start_idx + pos
                    draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
                    
                    found_idx = -1
                    target_prob_val = 0.0
                    
                    for k_i in range(compressed_vocab_size):
                        cand_idx = tl.load(target_indices_ptr + token_idx * compressed_vocab_size + k_i)
                        if cand_idx == draft_token_id:
                            found_idx = k_i
                            target_prob_val = tl.load(target_probs_ptr + token_idx * compressed_vocab_size + k_i)

                    if NO_DRAFT_PROBS:
                        draft_prob = 1.0
                    else:
                        draft_prob = tl.load(draft_probs_ptr + token_idx * compressed_vocab_size + draft_token_id) 
                        pass 
                    
                    uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                    
                    accept_threshold = target_prob_val / draft_prob
                    
                    if draft_prob > 0 and accept_threshold >= uniform_prob:
                        token_id = draft_token_id
                    else:
                        rejected = True
                        token_id = tl.load(recovered_token_ids_ptr + token_idx)
                    
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
            
            if not rejected:
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                    bonus_token_id,
                )
