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
 * \file reduce_common.h
 */
#ifndef REDUCE_COMMON_H_RMS_NORM
#define REDUCE_COMMON_H_RMS_NORM
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t MAX_REP_NUM = 255;
constexpr float ZERO = 0;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr int32_t HALf_INTERVAL = 2;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t INDEX_SIXTEEN = 16;
constexpr int32_t INDEX_EIGHT = 8;
constexpr static int32_t USED_UB_SIZE = 160 * 1024;

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

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline void CopyGMToGM_SplitBytes(
    AscendC::GlobalTensor<int8_t> &dst1,        // 第一部分目标（int8 视角）
    AscendC::GlobalTensor<int8_t> &dst2,        // 第二部分目标（int8 视角）
    AscendC::GlobalTensor<int8_t> &src,         // 源（连续字节）
    const uint32_t rowLen,
    const uint32_t rowTotalNum,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuf)
{
    constexpr uint32_t EVENT_ID0 = 0;
    constexpr uint32_t EVENT_ID1 = 1;
   
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    // 计算两段字节数（全部按 int8 字节计）
    int64_t part1Bytes = rowLen * rowTotalNum;          // 第一段字节数
    int64_t part2Bytes = rowTotalNum * sizeof(float);             // 第二段字节数
    int64_t totalBytes = part1Bytes + part2Bytes;

    // Ping-pong buffer 设置（以 int8_t 为单位）
    uint32_t tmpBufferLen = USED_UB_SIZE / 2;   // 单个 buffer 字节数
    constexpr int32_t BufferNum = 2;

    AscendC::LocalTensor<int8_t> ubBase = copyBuf.Get<int8_t>();
    AscendC::LocalTensor<int8_t> buf1 = ubBase;
    AscendC::LocalTensor<int8_t> buf2 = ubBase[tmpBufferLen];

    int pingpongId = 0;
    uint32_t ubMoveBytes = tmpBufferLen;
    auto processCount = CeilDiv(totalBytes, ubMoveBytes);

    for (uint32_t i = 0; i < processCount; ++i) {
        uint32_t curBytes = (i == processCount - 1)
                           ? totalBytes - i * ubMoveBytes
                           : ubMoveBytes;

        uint32_t copyBytes = static_cast<uint32_t>(curBytes);
        AscendC::TEventID eventId = (pingpongId == 0) ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int8_t> ub = (pingpongId == 0) ? buf1 : buf2;

        uint32_t srcOffset = i * ubMoveBytes;  // 当前块在 src 中的起始字节偏移

        // === Step 1: 等待 UB 可用 ===
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        // === Step 2: GM(src) -> UB ===
        AscendC::DataCopyPadExtParams<int8_t> padParams(false, 0, 0, 0);
        AscendC::DataCopyExtParams readParams(1, copyBytes, 0, 0, 0);
        AscendC::DataCopyPad(ub, src[srcOffset], readParams, padParams);
        // AscendC::DataCopy(ub, src[srcOffset], copyBytes);

        // === Step 3: 通知 UB 数据就绪 ===
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        // === Step 4: UB -> GM(dst1 或 dst2) ===
        if (srcOffset < part1Bytes) {
            // 当前块起始于 part1
            uint32_t part1End = part1Bytes;
            uint32_t blockEnd = srcOffset + curBytes;
            if (blockEnd <= part1End) {
                // 完全在 part1：写入 dst1
                AscendC::DataCopyExtParams writeParams(1, copyBytes, 0, 0, 0);
                AscendC::DataCopyPad(dst1[srcOffset], ub, writeParams);
            } else {
                // 跨越 part1/part2 边界
                uint32_t bytesInPart1 = part1End - srcOffset;
                uint32_t bytesInPart2 = curBytes - bytesInPart1;
                // 写 part1 部分到 dst1
                AscendC::DataCopyExtParams p1Params(1, bytesInPart1, 0, 0, 0);
                // AscendC::DataCopy(dst1[srcOffset], ub, bytesInPart1);
                AscendC::DataCopyPad(dst1[srcOffset], ub, p1Params);
                
                // // 写 part2 部分到 dst2（从 dst2[0] 开始）
                AscendC::DataCopyExtParams p2Params(1, bytesInPart2, 0, 0, 0);
                // AscendC::DataCopy(dst2[0], ub[bytesInPart1], bytesInPart2);
                AscendC::DataCopyPad(dst2[0], ub[bytesInPart1], p2Params);
            }
        } else {
            // 完全在 part2：写入 dst2
            uint32_t dst2Offset = srcOffset - part1Bytes;
            AscendC::DataCopyExtParams writeParams(1, copyBytes, 0, 0, 0);
            AscendC::DataCopyPad(dst2[dst2Offset], ub, writeParams);
        }

        // === Step 5: 释放 UB buffer ===
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        pingpongId = (pingpongId + 1) % BufferNum;
    }

    // 确保所有搬运完成
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t elemNum, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat1,
    const uint8_t repStride)
{
    uint32_t elemIndex = 0;
    for (; elemIndex + ELEM_PER_REP_FP32 <= numLastDim; elemIndex += ELEM_PER_REP_FP32) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, elemNum, repeat1,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, tailCount, repeat1,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
    }
    PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32); // set mask = 64
    WholeReduceSum<float, false>(dstLocal, tmpLocal, MASK_PLACEHOLDER, repeat1, 1, 1, ELEM_PER_BLK_FP32);
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim.
 */
__aicore__ inline void ReduceSumForSmallReduceDim(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t numLastDimAligned, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat,
    const uint8_t repStride)
{
    uint32_t smallRepeatTimes = repeat / MAX_REP_NUM;
    if (smallRepeatTimes == 0) {
        ReduceSumForSmallReduceDimPreRepeat(
            dstLocal, srcLocal, tmpLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat, repStride);
    } else {
        uint32_t smallRepTailNum = repeat % MAX_REP_NUM;
        uint32_t smallRepIndex = 0;
        uint32_t smallRepElem;
        for (; smallRepIndex + MAX_REP_NUM <= repeat; smallRepIndex += MAX_REP_NUM) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal[smallRepIndex], srcLocal[smallRepIndex * numLastDimAligned], tmpLocal[smallRepIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, MAX_REP_NUM, repStride);
        }
        if (smallRepTailNum != 0) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal[smallRepIndex], srcLocal[smallRepIndex * numLastDimAligned], tmpLocal[smallRepIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, smallRepTailNum, repStride);
        }
    }
}

__aicore__ inline int32_t findPowerTwo(int32_t n3)
{
    // find max power of 2 no more than n (32 bit)
    n3 |= n3 >> 1; // Set the first digit of n's binary to 1
    n3 |= n3 >> INDEX_TWO;
    n3 |= n3 >> INDEX_FOUR;
    n3 |= n3 >> INDEX_EIGHT;
    n3 |= n3 >> INDEX_SIXTEEN;
    return (n3 + 1) >> 1;
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim, require D < 255 * 8.
 * size of tmpLocal: (N, 64)
 */
__aicore__ inline void ReduceSumMultiN(
    const LocalTensor<float>& dstLocal2, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t numRow, const uint32_t numCol, const uint32_t numColAlign)
{
    const uint32_t tailCount = numCol % ELEM_PER_REP_FP32;
    const uint32_t repeat = numRow;
    const uint8_t repStride = numColAlign / ELEM_PER_BLK_FP32;
    Duplicate(tmpLocal, ZERO, numRow * ELEM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    ReduceSumForSmallReduceDim(dstLocal2, srcLocal, tmpLocal, numColAlign, numCol, tailCount, repeat, repStride);
}

__aicore__ inline void ReduceSumHalfInterval(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local3, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local3, src_local3, src_local3[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local3, src_local3, src_local3[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst_local, src_local3, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(dst_local, src_local3, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float>& src_local1, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local1, src_local1, src_local1[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local1, src_local1, src_local1[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(src_local1, src_local1, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(src_local1, src_local1, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    return src_local1.GetValue(0);
}
#endif // _REDUCE_COMMON_H_
