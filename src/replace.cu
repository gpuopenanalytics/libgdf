/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/replace.h>

#include <gdf/gdf.h>

namespace {

template <gdf_dtype DTYPE>
struct gdf_dtype_traits {};

#define DTYPE_FACTORY(DTYPE, T)                                                \
    template <>                                                                \
    struct gdf_dtype_traits<GDF_##DTYPE> {                                     \
        typedef T value_type;                                                  \
    }

DTYPE_FACTORY(INT8, std::int8_t);
DTYPE_FACTORY(INT16, std::int16_t);
DTYPE_FACTORY(INT32, std::int32_t);
DTYPE_FACTORY(INT64, std::int64_t);
DTYPE_FACTORY(FLOAT32, float);
DTYPE_FACTORY(FLOAT64, double);
DTYPE_FACTORY(DATE32, std::int32_t);
DTYPE_FACTORY(DATE64, std::int64_t);
DTYPE_FACTORY(TIMESTAMP, std::int64_t);

#undef DTYPE_FACTORY

template <class T>
__global__ void
replace_kernel(T *const                          data,
               const std::size_t                 data_size,
               const T *const                    values,
               const thrust::device_ptr<const T> to_replace_begin,
               const thrust::device_ptr<const T> to_replace_end) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < data_size;
         i += blockDim.x * gridDim.x) {
        // TODO: find by map kernel
        const thrust::device_ptr<const T> found_ptr = thrust::find(
          thrust::device, to_replace_begin, to_replace_end, data[i]);

        if (found_ptr != to_replace_end) {
            typename thrust::iterator_traits<
              const thrust::device_ptr<const T>>::difference_type
              value_found_index = thrust::distance(to_replace_begin, found_ptr);

            data[i] = values[value_found_index];
        }
    }
}

template <class T>
static inline gdf_error
Replace(T *const             data,
        const std::size_t    data_size,
        const T *const       to_replace,
        const T *const       values,
        const std::ptrdiff_t replacement_ptrdiff) {
    int multiprocessors;
    // TODO: device selection
    const cudaError_t status = cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, 0);

    if (status != cudaSuccess) { return GDF_CUDA_ERROR; }

    const std::size_t blocks = std::ceil(data_size / (multiprocessors * 256.));

    const thrust::device_ptr<const T> to_replace_begin(to_replace);
    const thrust::device_ptr<const T> to_replace_end(to_replace_begin
                                                     + replacement_ptrdiff);

    replace_kernel<T>
      <<<blocks * multiprocessors, 256>>>(  // TODO: calc blocks and threads
        data,
        data_size,
        values,
        to_replace_begin,
        to_replace_end);

    return GDF_SUCCESS;
}

static inline bool
NotEqualReplacementSize(const gdf_column *to_replace,
                        const gdf_column *values) {
    return to_replace->size != values->size;
}

static inline bool
NotSameDType(const gdf_column *column,
             const gdf_column *to_replace,
             const gdf_column *values) {
    return column->dtype != to_replace->dtype
           || to_replace->dtype != values->dtype;
}

}  // namespace

gdf_error
gdf_replace(gdf_column *      column,
            const gdf_column *to_replace,
            const gdf_column *values) {
    if (NotEqualReplacementSize(to_replace, values)) {
        return GDF_COLUMN_SIZE_MISMATCH;
    }

    if (NotSameDType(column, to_replace, values)) { return GDF_CUDA_ERROR; }

    switch (column->dtype) {
#define WHEN(DTYPE)                                                            \
    case GDF_##DTYPE: {                                                        \
        using value_type = gdf_dtype_traits<GDF_##DTYPE>::value_type;          \
        return Replace(static_cast<value_type *>(column->data),                \
                       static_cast<std::size_t>(column->size),                 \
                       static_cast<value_type *>(to_replace->data),            \
                       static_cast<value_type *>(values->data),                \
                       static_cast<std::ptrdiff_t>(values->size));             \
    }

        WHEN(INT8);
        WHEN(INT16);
        WHEN(INT32);
        WHEN(INT64);
        WHEN(FLOAT32);
        WHEN(FLOAT64);
        WHEN(DATE32);
        WHEN(DATE64);
        WHEN(TIMESTAMP);

#undef WHEN

    case GDF_invalid:
    default: return GDF_UNSUPPORTED_DTYPE;
    }
}
