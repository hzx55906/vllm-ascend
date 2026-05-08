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
 * \file add_rms_norm_dynamic_quant_ag_tiling.h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_AG_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_AG_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
// #include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling/tiling_base.h"
// #include "op_common/op_host/util/platform_util.h"
#include "tiling/tiling_templates_registry.h"
// #include "error_util.h"
#include "kernel_tiling/kernel_tiling.h"


namespace optiling {
typedef struct {    
    AscendC::tiling::Mc2InitTiling mc2InitTiling;
    AscendC::tiling::Mc2CcTiling mc2CcTiling;
    uint64_t useCore;             // 使用的 Core 数量
    uint64_t groupSize;             // group通信域大小
    uint64_t rowLen;
    uint64_t rowTotalNum;
    uint64_t numFirstDim;         // 第一维度的数量
    uint64_t numLastDimAligned;   // 对齐后的最后一维度数量
    uint64_t numLastDim;          // 最后一维度的数量
    uint64_t firstDimPerCoreTail; // 每个 Core 处理的第一维度尾部数量
    uint64_t firstDimPerCore;     // 每个 Core 处理的第一维度数量
    uint64_t firstDimPerLoop;     // 每次循环处理的第一维度数量
    uint64_t lastDimLoopNum;      // 最后一维度的循环次数
    uint64_t lastDimSliceLen;     // 最后一维度的切片长度
    uint64_t lastDimSliceLenTail; // 最后一维度的切片尾部长度
    uint32_t smoothNum1;          // 平滑参数 1
    uint32_t smoothNum2;          // 平滑参数 2
    float    epsilon;             // 防止除零的极小值
    int32_t  outQuant1Flag;       // 输出量化标志位 1
    int32_t  outQuant2Flag;       // 输出量化标志位 2
    float    avgFactor;           // 平均因子
    uint32_t betaFlag;            // Beta 标志位
} AddRmsNormDynamicQuantAGTilingData;

constexpr uint32_t TILING_TYPE_NORMAL = 0;
constexpr uint32_t TILING_TYPE_SPILT = 1;
constexpr uint32_t TILING_TYPE_PERF = 2;
constexpr uint32_t TILING_OFFSET_HAS_QUANT = 10;
constexpr uint32_t TILING_HAS_BETA = 40;
constexpr uint32_t TILING_OFFSET_REGBASE = 100;
constexpr uint64_t TILING_KEY_UNRUN = 199;

struct AddRmsNormDynamicQuantAGCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

enum class UB_TILING_POLICY : std::int32_t
{
    NORMAL,
    SINGLE_ROW,
    SLICE_D
};

static const gert::Shape g_vec_1_shape = {1};

inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

class AddRmsNormDynamicQuantAGTilingHelper {
public:
    explicit AddRmsNormDynamicQuantAGTilingHelper(gert::TilingContext* context) : context_(context)
    {}

    ~AddRmsNormDynamicQuantAGTilingHelper() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(AddRmsNormDynamicQuantAGTilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();
    bool CheckInputOutputShape();

    bool CheckUbNormalTiling();
    bool CheckUbSingleRowTiling();
    bool CheckUbSliceDTiling();
    bool ValidateBaseParameters();
    bool InitializePlatformInfo();
    bool ValidateInputOutput();
    bool CalculateShapeParameters();
    bool SetFlagsAndCheckConsistency();

    gert::TilingContext* context_;

    ge::DataType xDtype_{ge::DataType::DT_FLOAT16};
    uint64_t dtSize_{2};
    uint64_t socCoreNums_{1};
    uint64_t ubSize_{1};
    uint64_t sysWorkspaceSize_{1};

    uint64_t useCore_{1};
    uint64_t numFirstDim_{1};
    uint64_t numLastDim_{1};
    uint64_t numLastDimAligned_{1};
    uint64_t firstDimPerCore_{1};
    uint64_t firstDimPerCoreTail_{1};
    uint64_t firstDimPerLoop_{1};
    uint64_t lastDimSliceLen_{1};
    uint64_t lastDimLoopNum_{1};
    uint64_t lastDimSliceLenTail_{1};
    float eps_{1e-6};
    int32_t outQuant1Flag{0};
    int32_t outQuant2Flag{0};
    float avgFactor_{0.0};
    uint32_t smoothNum1_{0};
    uint32_t smoothNum2_{0};
    uint32_t betaFlag_{0};
    uint32_t dstType_{2};

    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::SINGLE_ROW};
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_H
