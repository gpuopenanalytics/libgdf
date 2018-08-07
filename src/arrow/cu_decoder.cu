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
#include "util/pinned_allocator.cuh"

namespace gdf
{
namespace arrow
{

CachingPinnedAllocator pinnedAllocator(2, 14, 29, 1024*1024*1024*1ull);

__host__ __device__ inline const uint32_t *unpack10_32(const uint32_t *in,
                                                       uint32_t *out)
{
    *out = ((*in) >> 0) % (1U << 10);
    out++;
    *out = ((*in) >> 10) % (1U << 10);
    out++;
    *out = ((*in) >> 20) % (1U << 10);
    out++;
    *out = ((*in) >> 30);
    ++in;
    *out |= ((*in) % (1U << 8)) << (10 - 8);
    out++;
    *out = ((*in) >> 8) % (1U << 10);
    out++;
    *out = ((*in) >> 18) % (1U << 10);
    out++;
    *out = ((*in) >> 28);
    ++in;
    *out |= ((*in) % (1U << 6)) << (10 - 6);
    out++;
    *out = ((*in) >> 6) % (1U << 10);
    out++;
    *out = ((*in) >> 16) % (1U << 10);
    out++;
    *out = ((*in) >> 26);
    ++in;
    *out |= ((*in) % (1U << 4)) << (10 - 4);
    out++;
    *out = ((*in) >> 4) % (1U << 10);
    out++;
    *out = ((*in) >> 14) % (1U << 10);
    out++;
    *out = ((*in) >> 24);
    ++in;
    *out |= ((*in) % (1U << 2)) << (10 - 2);
    out++;
    *out = ((*in) >> 2) % (1U << 10);
    out++;
    *out = ((*in) >> 12) % (1U << 10);
    out++;
    *out = ((*in) >> 22);
    ++in;
    out++;
    *out = ((*in) >> 0) % (1U << 10);
    out++;
    *out = ((*in) >> 10) % (1U << 10);
    out++;
    *out = ((*in) >> 20) % (1U << 10);
    out++;
    *out = ((*in) >> 30);
    ++in;
    *out |= ((*in) % (1U << 8)) << (10 - 8);
    out++;
    *out = ((*in) >> 8) % (1U << 10);
    out++;
    *out = ((*in) >> 18) % (1U << 10);
    out++;
    *out = ((*in) >> 28);
    ++in;
    *out |= ((*in) % (1U << 6)) << (10 - 6);
    out++;
    *out = ((*in) >> 6) % (1U << 10);
    out++;
    *out = ((*in) >> 16) % (1U << 10);
    out++;
    *out = ((*in) >> 26);
    ++in;
    *out |= ((*in) % (1U << 4)) << (10 - 4);
    out++;
    *out = ((*in) >> 4) % (1U << 10);
    out++;
    *out = ((*in) >> 14) % (1U << 10);
    out++;
    *out = ((*in) >> 24);
    ++in;
    *out |= ((*in) % (1U << 2)) << (10 - 2);
    out++;
    *out = ((*in) >> 2) % (1U << 10);
    out++;
    *out = ((*in) >> 12) % (1U << 10);
    out++;
    *out = ((*in) >> 22);
    ++in;
    out++;
    return in;
}

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

#define ARROW_CHECK(condition) \
    (condition) ? 0 : gdf::arrow::detail::CerrLog(ARROW_FATAL) << __FILE__ << __LINE__ << " Check failed: " #condition " "

#define DCHECK(condition) ARROW_CHECK(condition)
#define DCHECK_EQ(val1, val2) ARROW_CHECK((val1) == (val2))
#define DCHECK_NE(val1, val2) ARROW_CHECK((val1) != (val2))
#define DCHECK_LE(val1, val2) ARROW_CHECK((val1) <= (val2))
#define DCHECK_LT(val1, val2) ARROW_CHECK((val1) < (val2))
#define DCHECK_GE(val1, val2) ARROW_CHECK((val1) >= (val2))
#define DCHECK_GT(val1, val2) ARROW_CHECK((val1) > (val2))

/// Returns the 'num_bits' least-significant bits of 'v'.
__host__ __device__ static inline uint64_t TrailingBits(uint64_t v,
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
__host__ __device__ inline void GetValue_(int num_bits, T *v, int max_bytes,
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
            //
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
OutputIterator expand(InputIterator1 first1, InputIterator1 last1,
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

struct unpack10_32_functor
    : public thrust::binary_function<uint8_t, int, uint32_t>
{
    __host__ __device__ uint32_t operator()(uint8_t &input, int &output)
    {
        uint32_t *input_ptr = (uint32_t *)&input;
        uint32_t *output_ptr = (uint32_t *)&output;
        unpack10_32(input_ptr, output_ptr);
        return 0;
    }
};

typedef thrust::tuple<int, int, int, int> Int4;
struct remainder_functor : public thrust::unary_function<Int4, int>
{
    int max_bytes;
    int num_bits;
    uint8_t *buffer;
    int *ptr_output;
    remainder_functor(int max_bytes, int num_bits, uint8_t *buffer,
                      int *ptr_output)
        : max_bytes(max_bytes), num_bits(num_bits), buffer(buffer), ptr_output(ptr_output)
    {
    }
    __host__ __device__ int operator()(Int4 tuple)
    {
        int bit_offset = thrust::get<0>(tuple);  // remainderBitOffsets[k];
        int byte_offset = thrust::get<1>(tuple); // remainderInputOffsets[k];
        uint64_t buffered_values = 0;

        int bytes_remaining = max_bytes - byte_offset;
        if (bytes_remaining >= 8)
        {
            memcpy(&buffered_values, buffer + byte_offset, 8);
        }
        else
        {
            memcpy(&buffered_values, buffer + byte_offset, bytes_remaining);
        }
        int i = thrust::get<2>(tuple); // remainderOutputOffsets[k];
        int batch_size = thrust::get<2>(tuple) + thrust::get<3>(
                                                     tuple); // remainderOutputOffsets[k] + remainderSetSize[k];
        for (; i < batch_size; ++i)
        {
            detail::GetValue_(num_bits, &ptr_output[i], max_bytes, (uint8_t *)buffer,
                              &bit_offset, &byte_offset, &buffered_values);
        }
        return 0;
    }
};

int compute_step_size (const std::vector<int> &input_offset,
                       const std::vector<uint16_t> &is_rle = {}) 
{
    int step_size = 0; 
    for (int i = input_offset.size(); i > 0; i--) {
        step_size = input_offset[i]  - input_offset[i-1];
    }

    int bit_pack_count = 0;
    for (int i = 0; i < is_rle.size(); i++) {
        if (!is_rle[i]) {
            bit_pack_count++;
        }
    }
    return step_size;
}
//@todo: stream computing 
void gpu_bit_packing(const uint8_t *buffer, const int buffer_len,
                     const std::vector<int> &input_offset,
                     const std::vector<int> &output_offset,
                     thrust::device_vector<int>& d_output) 
{
    thrust::device_vector<int> d_output_offset(output_offset);
    
    // step_size in number of bytes
    int step_size = compute_step_size(input_offset);

    uint8_t* h_bit_buffer;
    pinnedAllocator.pinnedAllocate((void **)&h_bit_buffer, step_size * input_offset.size());

    thrust::device_vector<int> h_bit_offset(input_offset.size());
    for (int i = 0; i < input_offset.size(); i++) {
        memcpy(h_bit_buffer, &buffer[ input_offset[i] ], step_size);
        h_bit_offset[i] = i*step_size;
    }
    thrust::device_vector<uint8_t> d_bit_buffer(h_bit_buffer, h_bit_buffer + step_size * input_offset.size());
    thrust::device_vector<uint8_t> d_bit_offset(h_bit_offset);

    thrust::transform(thrust::cuda::par,
                        thrust::make_permutation_iterator(d_bit_buffer.begin(), d_bit_offset.begin()),
                        thrust::make_permutation_iterator(d_bit_buffer.end(), d_bit_offset.end()),
                        thrust::make_permutation_iterator(d_output.begin(), d_output_offset.begin()),
                        thrust::make_discard_iterator(), unpack10_32_functor());

}

int decode_using_gpu(const uint8_t *buffer, const int buffer_len,
                     const std::vector<uint32_t> &rle_runs,
                     const std::vector<uint64_t> &rle_values,
                     const std::vector<int> &input_offset,
                     const std::vector<int> &output_offset,
                     const std::vector<int> &remainderInputOffsets,
                     const std::vector<int> &remainderBitOffsets,
                     const std::vector<int> &remainderSetSize,
                     const std::vector<int> &remainderOutputOffsets,
                     const std::vector<uint16_t> &is_rle, int num_bits,
                     int *output, int batch_size)
{
    thrust::device_vector<uint8_t> d_buffer(buffer, buffer + buffer_len);
    thrust::device_vector<int> d_output(batch_size);
    thrust::device_vector<uint32_t> d_counts(rle_runs);
    thrust::device_vector<uint64_t> d_values(rle_values);

    gpu_expand(d_counts.begin(), d_counts.end(), d_values.begin(),
               d_output.begin());

    thrust::device_vector<int> d_remainder_input_offsets(remainderInputOffsets);
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
                     const std::vector<int> &output_offset,
                     const std::vector<int> &remainderInputOffsets,
                     const std::vector<int> &remainderBitOffsets,
                     const std::vector<int> &remainderSetSize,
                     const std::vector<int> &remainderOutputOffsets,
                     const std::vector<uint16_t> &is_rle, int num_bits,
                     int *output, int batch_size)
{
    thrust::host_vector<uint8_t> d_buffer(buffer, buffer + buffer_len);
    thrust::host_vector<int> d_output(batch_size);
    thrust::host_vector<uint32_t> d_counts(rle_runs);
    thrust::host_vector<uint64_t> d_values(rle_values);

    expand(d_counts.begin(), d_counts.end(), d_values.begin(), d_output.begin());

    thrust::device_vector<int> d_output_offset(output_offset);
    // step_size in number of bytes
    int step_size = compute_step_size(input_offset);

    thrust::host_vector<uint8_t> d_bitpacking_buffer( step_size * input_offset.size() );
    thrust::host_vector<int> d_input_bitpacking_offset;
    
    for (int i = 0; i < input_offset.size(); i++){
        memcpy ( &d_bitpacking_buffer[i*step_size] , &d_buffer[ input_offset[i] ], step_size );
        d_input_bitpacking_offset.push_back(i*step_size);
    }

    thrust::transform(
        thrust::host, thrust::make_permutation_iterator(d_bitpacking_buffer.begin(), d_input_bitpacking_offset.begin()),
        thrust::make_permutation_iterator(d_bitpacking_buffer.end(), d_input_bitpacking_offset.end()),
        thrust::make_permutation_iterator(d_output.begin(), d_output_offset.begin()),
        thrust::make_discard_iterator(), unpack10_32_functor());

    thrust::host_vector<int> d_remainder_input_offsets(remainderInputOffsets);
    thrust::host_vector<int> d_remainder_bit_offsets(remainderBitOffsets);
    thrust::host_vector<int> d_remainder_setsize(remainderSetSize);
    thrust::host_vector<int> d_remainder_output_offsets(remainderOutputOffsets);

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
        remainder_functor(max_bytes, num_bits, (uint8_t *)buffer, ptr_output));

    thrust::host_vector<int> host_output(d_output);
    for (int j = 0; j < batch_size; ++j)
    {
        output[j] = host_output[j];
    } 
    return batch_size;
}
} // namespace arrow
} // namespace gdf
