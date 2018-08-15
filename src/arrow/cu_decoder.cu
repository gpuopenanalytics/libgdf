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
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <iostream>
#include <tuple>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>

#include "cu_decoder.cuh"
#include "bpacking.cuh"
#include "util/pinned_allocator.cuh"

namespace gdf
{
namespace arrow
{
namespace internal {

CachingPinnedAllocator pinnedAllocator(2, 14, 29, 1024*1024*1024*1ull);
 
namespace detail
{

#define ARROW_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define ARROW_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))

#define ARROW_DEBUG (-1)
#define ARROW_INFO 0
#define ARROW_WARNING 1
#define ARROW_ERROR 2
#define ARROW_FATAL 3

class CerrLog
{
  public:
    CerrLog(int severity) // NOLINT(runtime/explicit)
        : severity_(severity),
          has_logged_(false)
    {
    }

    virtual ~CerrLog()
    {
        if (has_logged_)
        {
            std::cerr << std::endl;
        }
        if (severity_ == ARROW_FATAL)
        {
            std::exit(1);
        }
    }

    template <class T>
    CerrLog &operator<<(const T &t)
    {
        if (severity_ != ARROW_DEBUG)
        {
            has_logged_ = true;
            std::cerr << t;
        }
        return *this;
    }

  protected:
    const int severity_;
    bool has_logged_;
};
 

/// Returns the 'num_bits' least-significant bits of 'v'.
__device__  __host__  static inline uint64_t TrailingBits(uint64_t v,
                                                        int num_bits)
{
    if (ARROW_PREDICT_FALSE(num_bits == 0))
        return 0;
    if (ARROW_PREDICT_FALSE(num_bits >= 64))
        return v;
    int n = 64 - num_bits;
    return (v << n) >> n;
}

template <typename T>
__device__  __host__   inline void GetValue_(int num_bits, T *v, int max_bytes,
                                          const uint8_t *buffer,
                                          int *bit_offset, int *byte_offset,
                                          uint64_t *buffered_values)
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif
    *v = static_cast<T>(TrailingBits(*buffered_values, *bit_offset + num_bits) >> *bit_offset);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    *bit_offset += num_bits;

    if (*bit_offset >= 64)
    {
        *byte_offset += 8;
        *bit_offset -= 64;

        int bytes_remaining = max_bytes - *byte_offset;
        if (ARROW_PREDICT_TRUE(bytes_remaining >= 8))
        {
            memcpy(buffered_values, buffer + *byte_offset, 8);
        }
        else
        {
            memcpy(buffered_values, buffer + *byte_offset, bytes_remaining);
        }
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800 4805)
#endif
        // Read bits of v that crossed into new buffered_values_
        *v = *v | static_cast<T>(TrailingBits(*buffered_values, *bit_offset)
                                 << (num_bits - *bit_offset));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        // DCHECK_LE(*bit_offset, 64);
    }
}

} // namespace detail

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
OutputIterator gpu_expand(InputIterator1 first1, InputIterator1 last1,
                          InputIterator2 first2, OutputIterator output)
{
    typedef typename thrust::iterator_difference<InputIterator1>::type
        difference_type;

    difference_type input_size = thrust::distance(first1, last1);
    difference_type output_size = thrust::reduce(first1, last1);

    // scan the counts to obtain output offsets for each input element
    thrust::device_vector<difference_type> output_offsets(input_size, 0);
    thrust::exclusive_scan(first1, last1, output_offsets.begin());

    // scatter the nonzero counts into their corresponding output positions
    thrust::device_vector<difference_type> output_indices(output_size, 0);
    thrust::scatter_if(thrust::counting_iterator<difference_type>(0),
                       thrust::counting_iterator<difference_type>(input_size),
                       output_offsets.begin(), first1, output_indices.begin());

    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan(output_indices.begin(), output_indices.end(),
                           output_indices.begin(),
                           thrust::maximum<difference_type>());

    // gather input values according to index array (output =
    // first2[output_indices])
    OutputIterator output_end = output;
    thrust::advance(output_end, output_size);
    thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

    // return output + output_size
    thrust::advance(output, output_size);
    return output;
}

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
OutputIterator cpu_expand(InputIterator1 first1, InputIterator1 last1,
                      InputIterator2 first2, OutputIterator output)
{
    typedef typename thrust::iterator_difference<InputIterator1>::type
        difference_type;

    difference_type input_size = thrust::distance(first1, last1);
    difference_type output_size = thrust::reduce(first1, last1);

    // scan the counts to obtain output offsets for each input element
    thrust::host_vector<difference_type> output_offsets(input_size, 0);
    thrust::exclusive_scan(first1, last1, output_offsets.begin());

    // scatter the nonzero counts into their corresponding output positions
    thrust::host_vector<difference_type> output_indices(output_size, 0);
    thrust::scatter_if(thrust::counting_iterator<difference_type>(0),
                       thrust::counting_iterator<difference_type>(input_size),
                       output_offsets.begin(), first1, output_indices.begin());

    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan(output_indices.begin(), output_indices.end(),
                           output_indices.begin(),
                           thrust::maximum<difference_type>());

    // gather input values according to index array (output =
    // first2[output_indices])
    OutputIterator output_end = output;
    thrust::advance(output_end, output_size);
    thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

    // return output + output_size
    thrust::advance(output, output_size);
    return output;
}

__host__ __device__ inline const uint32_t* unpack32(const uint32_t* in, uint32_t* out, int num_bits) {
    const uint32_t* (*UnpackFunctionPtr[])(const uint32_t* in, uint32_t* out) = {nullunpacker32, unpack1_32, unpack2_32, unpack3_32, unpack4_32, unpack5_32, unpack6_32, unpack7_32, unpack8_32, unpack9_32, unpack10_32, unpack11_32, unpack12_32, unpack13_32, unpack14_32, unpack15_32, unpack16_32, unpack17_32, unpack18_32, unpack19_32, unpack20_32, unpack21_32, unpack22_32, unpack23_32, unpack24_32, unpack25_32, unpack26_32, unpack27_32, unpack28_32, unpack29_32, unpack30_32, unpack31_32, unpack32_32};
    return UnpackFunctionPtr[num_bits](in, out);
}

struct unpack_functor
    : public thrust::binary_function<uint8_t, int, uint32_t>
{
    int num_bits;
    unpack_functor(int num_bits) : num_bits(num_bits) {

    }
    __host__ __device__ uint32_t operator()(uint8_t &input, int &output)
    {
        uint32_t *input_ptr = (uint32_t *)&input;
        uint32_t *output_ptr = (uint32_t *)&output;
        
        unpack32(input_ptr, output_ptr, num_bits);

        return 0;
    }
};

typedef thrust::tuple<int, int, int, int> Int4;
struct remainder_functor : public thrust::unary_function<Int4, int>
{
    int max_bytes;
    int num_bits;
    uint8_t *d_buffer;
    int *ptr_output;
    remainder_functor(int max_bytes, int num_bits, uint8_t *buffer,
                      int *ptr_output)
        : max_bytes(max_bytes), num_bits(num_bits), d_buffer(buffer), ptr_output(ptr_output)
    {
    }
    __device__ __host__ int operator()(Int4 tuple)
    {
        int bit_offset = thrust::get<0>(tuple);  // remainderBitOffsets[k];
        int byte_offset = thrust::get<1>(tuple); // remainderInputOffsets[k];
        uint64_t buffered_values = 0;

        int bytes_remaining = max_bytes - byte_offset;
        if (bytes_remaining >= 8)
        {
            memcpy(&buffered_values, d_buffer + byte_offset, 8);
        }
        else
        {
            memcpy(&buffered_values, d_buffer + byte_offset, bytes_remaining);
        }
        int i = thrust::get<2>(tuple); // remainderOutputOffsets[k];
        int batch_size = thrust::get<2>(tuple) + thrust::get<3>(tuple); // remainderOutputOffsets[k] + remainderSetSize[k];
        for (; i < batch_size; ++i)
        {
            detail::GetValue_(num_bits, &ptr_output[i], max_bytes, (uint8_t *)d_buffer,
                              &bit_offset, &byte_offset, &buffered_values);
        }
        return 0;
    }
};

void gpu_bit_packing_remainder( const uint8_t *buffer,
                                const int buffer_len,
                                const std::vector<int> &remainderInputOffsets,
                                const std::vector<int> &remainderBitOffsets,
                                const std::vector<int> &remainderSetSize,
                                const std::vector<int> &remainderOutputOffsets,
                                thrust::device_vector<int>& d_output,
                                int num_bits) 
{
    int sum_set_size = 0;
    for (int i = 0; i < remainderInputOffsets.size(); i++){
        sum_set_size += (remainderSetSize[i] / 4 + 1) * 8;  
    }
    int offset = 0;
    thrust::host_vector<uint8_t> h_buffer(sum_set_size);
    thrust::host_vector<int> remainder_new_input_offsets;
    for (int i = 0; i < remainderInputOffsets.size(); i++) {
        auto offset_sz = (remainderSetSize[i] / 4 + 1) * 8;
        memcpy ( &h_buffer[offset], &buffer[ remainderInputOffsets[i] ], offset_sz);
        remainder_new_input_offsets.push_back(offset);
        offset += offset_sz;
    }
    thrust::device_vector<uint8_t> d_buffer(h_buffer);
    thrust::device_vector<int> d_remainder_input_offsets(remainder_new_input_offsets);
    thrust::device_vector<int> d_remainder_bit_offsets(remainderBitOffsets);
    thrust::device_vector<int> d_remainder_setsize(remainderSetSize);
    thrust::device_vector<int> d_remainder_output_offsets(remainderOutputOffsets);

    int max_bytes = buffer_len;
    auto zip_iterator_begin = thrust::make_zip_iterator(thrust::make_tuple(
        d_remainder_bit_offsets.begin(), d_remainder_input_offsets.begin(),
        d_remainder_output_offsets.begin(), d_remainder_setsize.begin()));
    auto zip_iterator_end = thrust::make_zip_iterator(thrust::make_tuple(
        d_remainder_bit_offsets.end(), d_remainder_input_offsets.end(),
        d_remainder_output_offsets.end(), d_remainder_setsize.end()));

    thrust::transform(
        thrust::device, zip_iterator_begin, zip_iterator_end,
        thrust::make_discard_iterator(),
        remainder_functor(max_bytes, num_bits, d_buffer.data().get(),
                          d_output.data().get()));
}

void cpu_bit_packing_remainder( const uint8_t *buffer,
                                const int buffer_len,
                                const std::vector<int> &remainderInputOffsets,
                                const std::vector<int> &remainderBitOffsets,
                                const std::vector<int> &remainderSetSize,
                                const std::vector<int> &remainderOutputOffsets,
                                thrust::host_vector<int>& d_output,
                                int num_bits)
{
    int sum_set_size = 0;
    for (int i = 0; i < remainderInputOffsets.size(); i++){
        sum_set_size += (remainderSetSize[i] / 4 + 1) * 8;  
    }
    int offset = 0;
    thrust::host_vector<uint8_t> h_buffer(sum_set_size);
    thrust::host_vector<int> remainder_new_input_offsets;
    for (int i = 0; i < remainderInputOffsets.size(); i++) {
        auto offset_sz = (remainderSetSize[i] / 4 + 1) * 8;
        memcpy ( &h_buffer[offset], &buffer[ remainderInputOffsets[i] ], offset_sz);
        remainder_new_input_offsets.push_back(offset);
        offset += offset_sz;
    }
    thrust::host_vector<uint8_t> &d_buffer(h_buffer);
    thrust::host_vector<int> &d_remainder_input_offsets(remainder_new_input_offsets);
    thrust::host_vector<int> &&d_remainder_bit_offsets(remainderBitOffsets);
    thrust::host_vector<int> &&d_remainder_setsize(remainderSetSize);
    thrust::host_vector<int> &&d_remainder_output_offsets(remainderOutputOffsets);

    int max_bytes = buffer_len;
    auto zip_iterator_begin = thrust::make_zip_iterator(thrust::make_tuple(
        d_remainder_bit_offsets.begin(), d_remainder_input_offsets.begin(),
        d_remainder_output_offsets.begin(), d_remainder_setsize.begin()));
    auto zip_iterator_end = thrust::make_zip_iterator(thrust::make_tuple(
        d_remainder_bit_offsets.end(), d_remainder_input_offsets.end(),
        d_remainder_output_offsets.end(), d_remainder_setsize.end()));
    int *ptr_output = &d_output[0];

    thrust::transform(
        thrust::host, zip_iterator_begin, zip_iterator_end,
        thrust::make_discard_iterator(),
        remainder_functor(max_bytes, num_bits, (uint8_t *)&d_buffer[0], ptr_output));
}

void cpu_bit_packing(const uint8_t *buffer, const int buffer_len,
                     const std::vector<int> &input_offset,
                     const std::vector<std::pair<uint32_t, uint32_t>>& bitpackset,
                     const std::vector<int> &output_offset,
                     thrust::host_vector<int>& d_output, 
                     int num_bits) 
{
    thrust::host_vector<int>&& d_output_offset(output_offset);
    int step_size = 32 * num_bits / 8;
    thrust::host_vector<uint8_t> h_bit_buffer( step_size * input_offset.size() );
    thrust::host_vector<int> h_bit_offset;

    for (int i = 0; i < input_offset.size(); i++){
        h_bit_offset.push_back(i*step_size);
    }
    int sum = 0;
    for (auto &&pair : bitpackset) {
       memcpy ( &h_bit_buffer[sum] , &buffer[pair.first], pair.second );
        sum += pair.second;
    }

    thrust::transform(
        thrust::host, thrust::make_permutation_iterator(h_bit_buffer.begin(), h_bit_offset.begin()),
        thrust::make_permutation_iterator(h_bit_buffer.end(), h_bit_offset.end()),
        thrust::make_permutation_iterator(d_output.begin(), d_output_offset.begin()),
        thrust::make_discard_iterator(), unpack_functor(num_bits));
}


//@todo: stream computing 
void gpu_bit_packing(const uint8_t *buffer, 
                     const int buffer_len,
                     const std::vector<int> &input_offset,
                     const std::vector<std::pair<uint32_t, uint32_t>>& bitpackset,
                     const std::vector<int> &output_offset,
                     thrust::device_vector<int>& d_output, 
                     int num_bits) 
{
    thrust::device_vector<int> d_output_offset(output_offset);
    int step_size = 32 * num_bits / 8;
    uint8_t* h_bit_buffer;
    pinnedAllocator.pinnedAllocate((void **)&h_bit_buffer, step_size * input_offset.size());

	thrust::host_vector<int> h_bit_offset;
	for (int i = 0; i < input_offset.size(); i++){
	    h_bit_offset.push_back(i*step_size);
	}
     int sum = 0;
    for (auto &&pair : bitpackset) {
	    memcpy ( &h_bit_buffer[sum] , &buffer[pair.first], pair.second );
        sum += pair.second;
	}
    thrust::device_vector<uint8_t> d_bit_buffer(h_bit_buffer, h_bit_buffer + step_size * input_offset.size());
    thrust::device_vector<int> d_bit_offset(h_bit_offset);

	thrust::transform(thrust::cuda::par,
	    thrust::make_permutation_iterator(d_bit_buffer.begin(), d_bit_offset.begin()),
	    thrust::make_permutation_iterator(d_bit_buffer.end(), d_bit_offset.end()),
	    thrust::make_permutation_iterator(d_output.begin(), d_output_offset.begin()),
	    thrust::make_discard_iterator(), unpack_functor(num_bits));
}

int decode_using_gpu(const uint8_t *buffer, const int buffer_len,
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
                     int *output, int batch_size)
{
    // thrust::device_vector<uint8_t> d_buffer(buffer, buffer + buffer_len);
    thrust::device_vector<int> d_output(batch_size);
    thrust::device_vector<uint32_t> d_counts(rle_runs);
    thrust::device_vector<uint64_t> d_values(rle_values);
 
    gpu_expand(d_counts.begin(), d_counts.end(), d_values.begin(),
               d_output.begin());

    gpu_bit_packing(buffer, buffer_len, input_offset, bitpackset, output_offset, d_output, num_bits);

    gpu_bit_packing_remainder(buffer, buffer_len, remainderInputOffsets, remainderBitOffsets, remainderSetSize, remainderOutputOffsets, d_output, num_bits);

    thrust::host_vector<int> host_output(d_output);
    for (int j = 0; j < batch_size; ++j)
    {
        output[j] = host_output[j];
    }
    return batch_size;
}


int decode_using_cpu(const uint8_t *buffer, const int buffer_len,
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
                     int *output, int batch_size)
{
    thrust::host_vector<int> d_output(batch_size);
    thrust::host_vector<uint32_t>&& d_counts(rle_runs);
    thrust::host_vector<uint64_t>&& d_values(rle_values);
    
    cpu_expand(d_counts.begin(), d_counts.end(), d_values.begin(), d_output.begin());

    cpu_bit_packing(buffer, buffer_len, input_offset, bitpackset, output_offset, d_output, num_bits);

    cpu_bit_packing_remainder(buffer, buffer_len, remainderInputOffsets, remainderBitOffsets, remainderSetSize, remainderOutputOffsets, d_output, num_bits);

    memcpy(output, &d_output[0], batch_size * sizeof (int) );
    return batch_size;
}
} // namespace internal
} // namespace arrow
} // namespace gdf
