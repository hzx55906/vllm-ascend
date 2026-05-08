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
 * \file add_rms_norm_dynamic_quant_ag_base.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_AG_BASE_CLASS_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_AG_BASE_CLASS_H_

#include "add_rms_norm_dynamic_quant_ag_helper.h"

constexpr static int32_t FLAG_OFFSET = 180 * 1024 * 1024;

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddRmsNormDynamicQuantAGBase {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantAGBase()
    {}

    __aicore__ inline void InitBaseParams(const AddRmsNormDynamicQuantAGTilingData* tiling)
    {
        this->groupSize = tiling->groupSize;
        this->rowLen = tiling->rowLen;
        this->rowTotalNum = tiling->rowTotalNum;
        this->numFirstDim = tiling->numFirstDim;
        this->numCore = tiling->useCore;
        this->numLastDimAligned = tiling->numLastDimAligned; // Quantize better be aligned to 32 elements
        this->numLastDim = tiling->numLastDim;

        this->firstDimPerCore = tiling->firstDimPerCore;
        this->firstDimPerCoreTail = tiling->firstDimPerCoreTail;
        this->firstDimPerLoop = tiling->firstDimPerLoop;

        this->lastDimLoopNum = tiling->lastDimLoopNum;
        this->lastDimSliceLen = tiling->lastDimSliceLen;
        this->lastDimSliceLenTail = tiling->lastDimSliceLenTail;
        this->betaFlag = tiling->betaFlag;
        this->aveNum = tiling->avgFactor;
        this->eps = tiling->epsilon;

        this->blockIdx_ = GetBlockIdx();
        if (this->blockIdx_ != this->numCore - 1) {
            this->rowStep = this->firstDimPerLoop;
            this->rowWork = this->firstDimPerCore;
        } else {
            this->rowWork = this->firstDimPerCoreTail;
            this->rowStep = TWO_NUMS_MIN(this->firstDimPerLoop, this->rowWork);
        }
        this->rowTail_ = (this->rowWork % this->rowStep == 0) ? this->rowStep : (this->rowWork % this->rowStep);
        this->gmOffset_ = this->firstDimPerCore * this->numLastDim;

        this->smooth1Exist = tiling->smoothNum1;
        // 2 dynamic quant operator required 2 scale buffer.
        this->smooth2Exist = tiling->smoothNum2;

        // dynamic quant max value
        if constexpr (IsSameType<T_Y, int8_t>::value) {
            this->quantMaxVal = DYNAMIC_QUANT_DIVIDEND;
        } else {
            this->quantMaxVal = DYNAMIC_QUANT_DIVIDEND_INT4;
        }
        this->outQuant1Flag = tiling->outQuant1Flag;
        this->outQuant2Flag = tiling->outQuant2Flag;

        this->isOld = (this->outQuant1Flag == -1) && (this->outQuant2Flag == -1);
        this->oldDouble = this->isOld && this->smooth1Exist && this->smooth2Exist;
        this->newSingleFirst = this->smooth1Exist && (this->outQuant1Flag == 1);
        this->newSingleSecond = this->smooth2Exist && (this->outQuant2Flag == 1);
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        
        this->hccl_.InitV2(contextGM0, tiling);
        this->hccl_.SetCcTilingV2(offsetof(AddRmsNormDynamicQuantAGTilingData, mc2CcTiling));
        for (int i = 0; i < tiling->groupSize; i++) {
            this->buff[i] = (GM_ADDR)this->hccl_.GetWindowsInAddr(i);
        }
        this->rankId = this->hccl_.GetRankId();
    }

    __aicore__ inline void InitInGlobalTensors(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR beta)
    {
        x1Gm.SetGlobalBuffer((__gm__ T*)(x1) + blockIdx_ * this->gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T*)(x2) + blockIdx_ * this->gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
        smooth1Gm.SetGlobalBuffer((__gm__ T*)smooth1);
        smooth2Gm.SetGlobalBuffer((__gm__ T*)smooth2);
        if (this->betaFlag == 1) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta);
        }
    }

    __aicore__ inline void InitOutGlobalTensors(GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2)
    {
        int64_t yBufferSize = blockIdx_ * this->gmOffset_;
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            yBufferSize = yBufferSize / 2;
        }
        y1Gm.SetGlobalBuffer((__gm__ T_Y*)(y1) + yBufferSize);
        y2Gm.SetGlobalBuffer((__gm__ T_Y*)(y2) + yBufferSize);
        xGm.SetGlobalBuffer((__gm__ T*)(x) + blockIdx_ * this->gmOffset_);
        outScale2Gm.SetGlobalBuffer((__gm__ float*)outScale2 + blockIdx_ * this->firstDimPerCore);
        outScale1Gm.SetGlobalBuffer((__gm__ float*)outScale1 + blockIdx_ * this->firstDimPerCore);
    }

    __aicore__ inline void InitWorkSpaceGlobalTensors(GM_ADDR workspace)
    {}

      __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
    {
        if (blockIdx_ == 0) {
            SetBuffFlag((__gm__ int32_t *)(buff[this->rankId] + FLAG_OFFSET + flag_idx*sizeof(int32_t)), flag_data);
        }
        if (blockIdx_ < this->groupSize) {
            CheckBuffFlag((__gm__ int32_t *)(buff[blockIdx_] + FLAG_OFFSET + flag_idx*sizeof(int32_t)), flag_data);
        }
    }
    template <typename TC>
    __aicore__ inline void CopyUbufToGmAlignB16(__gm__ TC *dst, LocalTensor<TC> ubTensor, uint16_t nBurst, uint32_t lenBurst,
                                                uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,     // blockCount
                                        lenBurst,   // blockLen
                                        srcStride,  // srcStride
                                        dstStride,  // dstStride
                                        0);
        GlobalTensor<TC> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }
    template <typename TC>
    __aicore__ inline void CopyGmToUbufAlignB16(LocalTensor<TC> ubTensor, __gm__ TC *src, uint16_t nBurst, uint32_t lenBurst,
                                                uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,     // blockCount
                                        lenBurst,   // blockLen
                                        srcStride,  // srcStride
                                        dstStride,  // dstStride
                                        0);
        GlobalTensor<TC> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopyPadExtParams<TC> padParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
    }

    __aicore__ inline void SetBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        LocalTensor<int32_t> ubTensor = flagBuf.Get<int32_t>();
        ubTensor(0) = flag;
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    }

    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        LocalTensor<int32_t> ubTensor = flagBuf.Get<int32_t>();
        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) == flag) {
                break;
            }
        }
    }

    template <pipe_t pipe>
    inline __aicore__ void FFTSCrossCoreSync(uint64_t mode, uint64_t flag_id)
    {
        uint64_t config = 1 | (mode << 4) | (flag_id << 8);
        ffts_cross_core_sync(pipe, config);
    }
    __aicore__ inline void SetAndWaitAivSync(uint64_t flag_idx, int32_t pipe_depth = 2)
    {
        FFTSCrossCoreSync<PIPE_MTE3>(0, flag_idx + pipe_depth);
        WaitEvent(flag_idx + pipe_depth);
    }
    __aicore__ inline void ResetIpcFlags(int32_t num_flags)
    {
        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (blockIdx_ == 0){
                SetBuffFlag((__gm__ int32_t *)(buff[this->rankId] + FLAG_OFFSET + idx*sizeof(int32_t)), 0);
            }
        }
    }
protected:
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    GM_ADDR buff[16];
    GM_ADDR y1Out;
    GM_ADDR scale1Out;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copyBuf, flagBuf;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> smooth1Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> smooth2Gm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T_Y> y1Gm;
    GlobalTensor<T_Y> y2Gm;
    GlobalTensor<T> xGm;
    GlobalTensor<float> outScale1Gm;
    GlobalTensor<float> outScale2Gm;

    uint32_t betaFlag;
    uint64_t numFirstDim;
    uint64_t numCore;
    uint64_t numLastDim;
    uint64_t firstDimPerCore;
    uint64_t numLastDimAligned;
    uint64_t firstDimPerCoreTail;
    uint64_t firstDimPerLoop;
    uint64_t lastDimLoopNum;
    uint64_t lastDimSliceLen;
    uint64_t lastDimSliceLenTail;

    float aveNum;
    float eps;

    uint64_t gmOffset_;
    uint64_t blockIdx_;
    uint64_t rowTail_;
    uint64_t rowStep;
    uint64_t rowWork;

    bool smooth1Exist;
    bool smooth2Exist;
    int32_t outQuant1Flag;
    int32_t outQuant2Flag;
    
    bool isOld;
    bool oldDouble;
    bool newSingleFirst;
    bool newSingleSecond;
    float quantMaxVal;
    uint32_t rankId;
    uint32_t groupSize;
    uint32_t rowLen;
    uint32_t rowTotalNum;
};

#endif // __ADD_RMS_NORM_DYNAMIC_QUANT_BASE_CLASS_H_
