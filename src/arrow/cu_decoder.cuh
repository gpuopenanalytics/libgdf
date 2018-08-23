#ifndef _CU_DECODER_H_
#define _CU_DECODER_H_
/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *     Copyright 2018 William Malpica <william@blazingdb.com>
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

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace gdf {
namespace arrow {
namespace internal {
    
    template<typename T>
    int decode_using_gpu(const T *dictionary, int num_dictionary_values, T* d_output, const uint8_t *buffer, const int buffer_len,
                     const std::vector<uint32_t> &rle_runs,
                     const std::vector<uint64_t> &rle_values,
                     const std::vector<int> &input_offset,
                     const std::vector<std::pair<uint32_t, uint32_t>>& bitpackset,
                     const std::vector<int> &output_offset,
                     const std::vector<int> &remainderInputOffsets,
                     const std::vector<int> &remainderBitOffsets,
                     const std::vector<int> &remainderSetSize,
                     const std::vector<int> &remainderOutputOffsets,
                     const std::vector<uint16_t> &is_rle, int num_bits,
                     int batch_size);

    template<typename T>
    int unpack_using_gpu(const uint8_t* buffer, const int buffer_len,
                 const std::vector<int>& input_offset,
                 const std::vector<std::pair<uint32_t, uint32_t>>& bitpackset,
                 const std::vector<int>& output_offset,
                 const std::vector<int>& remainderInputOffsets,
                 const std::vector<int>& remainderBitOffsets,
                 const std::vector<int>& remainderSetSize,
                 const std::vector<int>& remainderOutputOffsets,
                 int num_bits,
                 T* output, int batch_size);


    // expands data vector that does not contain nulls into a representation that has indeterminate values where there should be nulls
    // The expansion happens in place. This assumes that the data vector is actually big enough to hold the expanded data
    // A vector of int work_space needs to be allocated to hold the prefix sum.
    template <typename T>
    void compact_to_sparse_for_nulls(T* data, const uint8_t* valid_bits, int batch_size, int * work_space){
    	thrust::exclusive_scan	(valid_bits, valid_bits + batch_size, work_space);

    	thrust::gather_if(work_space, work_space + batch_size, valid_bits, data, data);
    }


    template<typename Func>
    __global__
    void decode_bitpacking(uint8_t *buffer, int *output, int *input_offsets, int *input_run_lengths,
    		int * output_offsets, int *output_run_lengths, short bit_width, int max_run_length, Func unpack_func)
    {

    	short INPUT_BLOCK = bit_width * 32 / 8; // number of bytes needed for a unpack32 operation
    	short OUTPUT_BLOCK = 32; // number of elements for output

    	int index = blockIdx.x * blockDim.x + threadIdx.x;

    	int set_index = index/max_run_length;
    	int intput_index = input_offsets[set_index] + INPUT_BLOCK * (index % max_run_length);

    	if ((INPUT_BLOCK * (index % max_run_length)) < input_run_lengths[set_index]) { // if we want to actually process

    		uint8_t temp_in[INPUT_BLOCK];
    		int temp_out[OUTPUT_BLOCK];

    		for (int i = 0; i < INPUT_BLOCK; i++){
    			temp_in[i] = buffer[intput_index + i];
    		}
    		unpack_func(temp_in, temp_out);

    		for (int i = 0; i < INPUT_BLOCK; i++){
    			output[output_index + i] = temp_out[i];
    		}
    	}
    }


}
} // namespace arrow
} // namespace gdf

#endif // _CU_DECODER_H_
