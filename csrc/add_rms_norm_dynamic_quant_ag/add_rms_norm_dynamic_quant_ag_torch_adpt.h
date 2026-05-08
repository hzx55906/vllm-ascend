/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_AG_ADPT_H
#define ADD_RMS_NORM_DYNAMIC_QUANT_AG_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> add_rms_norm_dynamic_quant_ag(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const c10::optional<at::Tensor> & smooth_scale1, const c10::optional<at::Tensor> & smooth_scale2, const c10::optional<at::Tensor> & beta, c10::string_view group, int64_t group_size, double epsilon, ::std::array<bool,2> output_mask)
{
    auto group_ptr = const_cast<char *>(group.data());
    auto output_size_0 = ([&]() { std::vector<int64_t> v(x1.sizes().begin(), x1.sizes().end()); v[0] *= group_size; return v; })();
    auto output_size_1 = output_mask[1] ? x1.sizes() : at::IntArrayRef{};
    auto output_size_2 = x1.sizes();
    auto output_size_3 = ([&]() { std::vector<int64_t> v(x1.sizes().begin(), x1.sizes().end()); v[0] *= group_size; v.pop_back(); return v; })();
    auto output_size_4 = output_mask[1] ? output_size_3 : std::vector<int64_t>{};
    auto output_dtype_0 = at::kChar;
    auto output_dtype_1 = x1.scalar_type();
    auto output_dtype_2 = at::kFloat;
    at::Tensor y1 = at::empty(output_size_0, x1.options().dtype(output_dtype_0));
    at::Tensor y2 = at::empty(output_size_1, x1.options().dtype(output_dtype_0));
    at::Tensor x_out = at::empty(output_size_2, x1.options().dtype(output_dtype_1));
    at::Tensor scale1 = at::empty(output_size_3, x1.options().dtype(output_dtype_2));
    at::Tensor scale2 = at::empty(output_size_4, x1.options().dtype(output_dtype_2));
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantAG, x1, x2, gamma, smooth_scale1, smooth_scale2, beta, group_ptr, group_size, epsilon, output_mask, y1, y2, x_out, scale1, scale2);
    return std::make_tuple(std::move(y1), std::move(y2), std::move(x_out), std::move(scale1), std::move(scale2));
}
}
#endif