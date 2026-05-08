import gc
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import torchair

from vllm_ascend.utils import enable_custom_op

config = torchair.CompilerConfig()
config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)
torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

global_rank_id = 0


def golden_op_add_rms_norm_quant_all_gather(x1, x2, gamma, ep_world_size, rank):
    x_normed, _, _ = torch_npu.npu_add_rms_norm(
        x1, 
        x2, 
        gamma, 
    ) 
    gathertensor = torch.empty(ep_world_size * x_normed.shape[0], x_normed.shape[1], dtype=torch.float16, device=f"npu:{rank}")
    dist.all_gather_into_tensor(gathertensor, x_normed)
    
    quant, scale = torch_npu.npu_dynamic_quant(gathertensor, dst_type=torch.int8)
    return quant, scale


def worker(rank, ep_world_size, batch_size, m, k, n):
    global global_rank_id
    global_rank_id = rank
    rank = rank

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="hccl",
                            rank=rank,
                            world_size=ep_world_size)

    ep_ranks_list = list(np.arange(0, ep_world_size))

    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)

    torch_npu.npu.set_device(rank)
    ep_hcomm_info = ep_group._get_backend(
        torch.device("npu")).get_hccl_comm_name(rank)

    torch_npu.npu.synchronize(rank)

    class Module(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x1, x2, gamma, ep_hcomm_info, output_mask):
            y1, y2, x_out, s1, s2 = torch.ops._C_ascend.add_rms_norm_dynamic_quant_ag(
                x1, 
                x2, 
                gamma, 
                group=ep_hcomm_info, 
                group_size=ep_world_size, 
                output_mask=output_mask,
            )
            return y1, s1

    DTYPE = torch.float16

    torch.manual_seed(42)

    x1 = torch.ones([m, k], dtype=DTYPE).npu(rank)
    x2 = torch.ones([n, k], dtype=DTYPE).npu(rank)

    output_mask = [True, False]
    gamma = torch.full([k], 1, dtype=DTYPE).npu(rank)

    epsilon = 1e-5
    warnup_cnt = 5
    repeat_cnt = 10

    def run_golden_case(loop_cnt):
        for _ in range(loop_cnt):
            golden_quant, golden_scale = golden_op_add_rms_norm_quant_all_gather(x1, x2, gamma, ep_world_size, rank)
        torch_npu.npu.synchronize(rank)
        return golden_quant, golden_scale

    run_golden_case(warnup_cnt)

    golden_quant, golden_scale = run_golden_case(repeat_cnt)
    golden_quant = golden_quant.detach().cpu()
    golden_scale = golden_scale.detach().cpu()

    mod = Module().npu()
    opt_model = torch.compile(mod, backend=npu_backend)

    def run_custom_case(loop_cnt):
        for _ in range(loop_cnt):
            quant, scale = opt_model(x1, x2, gamma, ep_hcomm_info, output_mask)
        torch_npu.npu.synchronize(rank)
        return quant, scale

    # warn up
    run_custom_case(warnup_cnt)

    quant, scale = run_custom_case(repeat_cnt)
    quant = quant.detach().cpu()
    scale = scale.detach().cpu()

    dist.destroy_process_group()

    torch.testing.assert_close(golden_quant, quant, atol=0.1, rtol=0.005)
    torch.testing.assert_close(golden_scale, scale, atol=0.1, rtol=0.005)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_add_rms_norm_quant_all_gather_kernel():
    ep_world_size = 4
    batch_size = 1
    m = 64
    k = 3072
    n = 64
    args = (ep_world_size, batch_size, m, k, n)
    mp.spawn(worker, args=args, nprocs=ep_world_size, join=True)
