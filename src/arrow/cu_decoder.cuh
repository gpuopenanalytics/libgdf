#ifndef _CU_DECODER_H_
#define _CU_DECODER_H_
/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace gdf {
namespace arrow {

    int decode_using_cpu(const uint8_t* buffer, const int buffer_len,
        const std::vector<uint32_t>& rle_runs,
        const std::vector<uint64_t>& rle_values,
        const std::vector<int>& input_offset,
        const std::vector<int>& output_offset,
        const std::vector<int>& remainderInputOffsets,
        const std::vector<int>& remainderBitOffsets,
        const std::vector<int>& remainderSetSize,
        const std::vector<int>& remainderOutputOffsets,
        const std::vector<uint16_t>& is_rle, int num_bits,
        int* output, int batch_size);

    int decode_using_gpu(const uint8_t* buffer, const int buffer_len,
        const std::vector<uint32_t>& rle_runs,
        const std::vector<uint64_t>& rle_values,
        const std::vector<int>& input_offset,
        const std::vector<int>& output_offset,
        const std::vector<int>& remainderInputOffsets,
        const std::vector<int>& remainderBitOffsets,
        const std::vector<int>& remainderSetSize,
        const std::vector<int>& remainderOutputOffsets,
        const std::vector<uint16_t>& is_rle, int num_bits,
        int* output, int batch_size);

} // namespace arrow
} // namespace gdf

#endif // _CU_DECODER_H_
