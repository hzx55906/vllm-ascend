import numpy as np
import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed import tensor_model_parallel_all_gather

def compute_logits(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    logits = self.logits_processor(self.lm_head, hidden_states)
    if self.draft_id_to_target_id is None:
        assert logits.shape[1] == self.config.vocab_size, (
            "Expected logits to have shape "
            f"(*, {self.config.vocab_size}), but got {logits.shape}"
        )
        return logits
    # print("logits0:",logits.shape)
    logits = logits.contiguous()
    next_token = greedy_sample(logits)
    # print("next_token0", next_token.shape)
    bias = torch.index_select(self.draft_id_to_target_id, dim=0, index=next_token.view(-1)).view(next_token.shape)
    # tp_group = get_tp_group()
    # logits = tp_group.all_gather(logits, -1)
    # print("logits1:", logits.shape)
    # logits = tensor_model_parallel_all_gather(logits, -1)
    # base = torch.arange(self.config.draft_vocab_size, device=logits.device)
    # targets = base + self.draft_id_to_target_id
    logits_new = logits.new_full(
        (
            logits.shape[0],
            self.config.vocab_size,
        ),
        float("-inf"),
    )
    # logits_new[:, targets] = logits
    # logits_new = logits.new_full(
    #     (
    #         logits.shape[0],
    #         self.config.vocab_size,
    #     ),
    #     float("-inf"),
    # )
    # logits_new[:, targets] = logits
    # return logits_new
    return logits_new, next_token + bias

# def compute_logits(
#         self,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor | None:
#         logits = self.logits_processor(self.lm_head, hidden_states)
#         if self.draft_id_to_target_id is None:
#             assert logits.shape[1] == self.config.vocab_size, (
#                 "Expected logits to have shape "
#                 f"(*, {self.config.vocab_size}), but got {logits.shape}"
#             )
#             return logits

#         base = torch.arange(self.config.draft_vocab_size, device=logits.device)
#         targets = base + self.draft_id_to_target_id
#         logits_new = logits.new_full(
#             (
#                 logits.shape[0],
#                 self.config.vocab_size,
#             ),
#             float("-inf"),
#         )
#         logits_new[:, targets] = logits
#         return logits_new
    
def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    tp_group = get_tp_group()
    B, V_local = logits.shape
    rank = tp_group.rank_in_group
    world_size = tp_group.world_size

    local_max_logits, local_max_indices = logits.max(dim=-1)

    # print("local_max_logits", local_max_logits.shape)
    # print("local_max_indices", local_max_indices.shape)
    local_global_idx = local_max_indices + rank * V_local  # [B]

    # [B, world_size]
    gathered_logits = tp_group.all_gather(local_max_logits, dim=-1).view(B, world_size)

    gathered_global_idx = tp_group.all_gather(local_global_idx, dim=-1).view(B, world_size)  # [B, world_size]
    # print("gathered_logits", gathered_logits.shape)
    # print("gathered_global_idx", gathered_global_idx.shape)
    global_max_rank = gathered_logits.argmax(dim=-1)  # [B]
    # print("global_max_rank", global_max_rank.shape)
    target_argmax = gathered_global_idx.gather(
        dim=-1,
        index=global_max_rank.unsqueeze(-1)
    ).squeeze(-1) # [B]
    # print("target_argmax", target_argmax.shape)
    return target_argmax

Eagle3LlamaForCausalLM.compute_logits = compute_logits
