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
 * \file fallback_matmul_all_reduce.cpp
 * \brief
 */
#include "fallback.h"

namespace fallback {

static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int SMOOTH1_IDX = 3;
static constexpr int SMOOTH2_IDX = 4;
static constexpr int BETA_IDX = 5;

static constexpr int Y1_IDX = 0;
static constexpr int Y2_IDX = 1;
static constexpr int X_IDX = 2;
static constexpr int OUT_SCALE1_IDX = 3;
static constexpr int OUT_SCALE2_IDX = 4;

static constexpr int GROUP_IDX = 0;
static constexpr int GROUP_SIZE_IDX = 1;
static constexpr int EPS_IDX = 2;
static constexpr int OUT_QUANT_1_IDX = 3;

inline const char* AddRmsNormDynamicQuantAGInfo = "AddRmsNormDynamicQuantAGFallback";

ge::graphStatus AddRmsNormDynamicQuantAGExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D(AddRmsNormDynamicQuantAGInfo, "Start to fallback for allgather matmul.");
  OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);
  
  const auto x1 = host_api_ctx->GetInputTensor(static_cast<size_t>(X1_IDX));
  OPS_ERR_IF(x1 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "x1 is null"), return ge::GRAPH_FAILED);

  const auto x2 = host_api_ctx->GetInputTensor(static_cast<size_t>(X2_IDX));
  OPS_ERR_IF(x2 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "x2 is null"), return ge::GRAPH_FAILED);
  
  const auto gamma = host_api_ctx->GetInputTensor(static_cast<size_t>(GAMMA_IDX));
  OPS_ERR_IF(gamma == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "gamma is null"), return ge::GRAPH_FAILED);

  const auto smooth_scale1 = host_api_ctx->GetInputTensor(static_cast<size_t>(SMOOTH1_IDX));
//   OPS_ERR_IF(smooth_scale1 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "smooth_scale1 is null"), return ge::GRAPH_FAILED);

  const auto smooth_scale2 = host_api_ctx->GetInputTensor(static_cast<size_t>(SMOOTH2_IDX));
//   OPS_ERR_IF(smooth_scale2 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "smooth_scale2 is null"), return ge::GRAPH_FAILED);

  const auto beta = host_api_ctx->GetInputTensor(static_cast<size_t>(BETA_IDX));


  const auto y1 = host_api_ctx->GetOutputTensor(static_cast<size_t>(Y1_IDX));
  OPS_ERR_IF(y1 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "y1 is null"), return ge::GRAPH_FAILED);

  const auto y2 = host_api_ctx->GetOutputTensor(static_cast<size_t>(Y2_IDX));
  OPS_ERR_IF(y2 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "y2 is null"), return ge::GRAPH_FAILED);

  const auto x_out = host_api_ctx->GetOutputTensor(static_cast<size_t>(X_IDX));
  OPS_ERR_IF(x_out == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "x_out is null"), return ge::GRAPH_FAILED);

  const auto scale1 = host_api_ctx->GetOutputTensor(static_cast<size_t>(OUT_SCALE1_IDX));
  OPS_ERR_IF(scale1 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "scale1 is null"), return ge::GRAPH_FAILED);

  const auto scale2 = host_api_ctx->GetOutputTensor(static_cast<size_t>(OUT_SCALE2_IDX));
  OPS_ERR_IF(scale2 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "scale2 is null"), return ge::GRAPH_FAILED);

  
  const auto attrs = host_api_ctx->GetAttrs();
  OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "attrs is null"), return ge::GRAPH_FAILED);

  const char *group_ptr = attrs->GetStr(static_cast<size_t>(GROUP_IDX));
  OPS_ERR_IF(group_ptr == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "group is null"), return ge::GRAPH_FAILED);

  const int64_t *group_size = attrs->GetInt(static_cast<size_t>(GROUP_SIZE_IDX));
  OPS_ERR_IF(group_size == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "group_size is null"), return ge::GRAPH_FAILED);

  const auto epsilon = attrs->GetFloat(static_cast<size_t>(EPS_IDX));
  OPS_ERR_IF(epsilon == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "epsilon is null"), return ge::GRAPH_FAILED);
  
  const auto output_mask = attrs->GetAttrPointer<gert::ContinuousVector>(OUT_QUANT_1_IDX);;
  OPS_ERR_IF(output_mask == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "output_mask is null"), return ge::GRAPH_FAILED);

  const auto api_ret = EXEC_OPAPI_CMD(aclnnAddRmsNormDynamicQuantAG, x1, x2, gamma, smooth_scale1, smooth_scale2, beta, group_ptr, group_size, epsilon, output_mask, y1, y2, x_out, scale1, scale2);
  OPS_ERR_IF(api_ret != ge::GRAPH_SUCCESS, OPS_LOG_E(AddRmsNormDynamicQuantAGInfo, "Aclnn api error code %u", api_ret),
           return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(AddRmsNormDynamicQuantAG).OpExecuteFunc(AddRmsNormDynamicQuantAGExecuteFunc);
} // namespace fallback
