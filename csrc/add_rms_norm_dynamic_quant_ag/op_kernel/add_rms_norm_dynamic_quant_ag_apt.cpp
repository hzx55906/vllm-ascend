/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file add_rms_norm_dynamic_quant_ag_apt.cpp
 * \brief
 */

#include "add_rms_norm_dynamic_quant_ag_normal_kernel.h"
#include "add_rms_norm_dynamic_quant_ag_single_row_kernel.h"
#include "add_rms_norm_dynamic_quant_ag_cut_d_kernel.h"



using namespace AscendC;

#define TILING_KEY_UNRUN 199

#define INIT_AND_PROCESS_WORKSPACE                                                                                 \
    do {                                                                                                           \
        op.Init(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, scale1, scale2, usrWorkspace, tilingData); \
        op.Process();                                                                                              \
    } while (0)



extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant_ag(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2, GM_ADDR beta, GM_ADDR y1,
    GM_ADDR y2, GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicQuantAGTilingData, tilingDataIn, tiling);
    const AddRmsNormDynamicQuantAGTilingData* __restrict tilingData = &tilingDataIn;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(0)) {
        // 0 Tiling, Do Nothing.
    } else if (TILING_KEY_IS(1)) {
        KernelAddRmsNormDynamicQuantAGNormal<DTYPE_X1, DTYPE_Y1, 1> op(&pipe);
        INIT_AND_PROCESS_WORKSPACE;
    } else if (TILING_KEY_IS(2)) {
        KernelAddRmsNormDynamicQuantAGSingleRow<DTYPE_X1, DTYPE_Y1, 2> op(&pipe);
        INIT_AND_PROCESS_WORKSPACE;
    } else if (TILING_KEY_IS(3)) {
        KernelAddRmsNormDynamicQuantAGSliceD<DTYPE_X1, DTYPE_Y1, 3> op(&pipe);
        INIT_AND_PROCESS_WORKSPACE;
    }
}
