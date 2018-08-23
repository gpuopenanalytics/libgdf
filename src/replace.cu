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

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include <gdf/gdf.h>

namespace {

template <gdf_dtype DTYPE>
struct gdf_dtype_traits {};

#define DTYPE_FACTORY(DTYPE, T)                                               \
    template <>                                                               \
    struct gdf_dtype_traits<GDF_##DTYPE> {                                    \
        typedef T value_type;                                                 \
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
class ReplaceFunctor {
public:
    ReplaceFunctor(T *const raw) : data(raw) {}

    __host__ __device__ void
             operator()(thrust::tuple<const T, const T, const T> tuple) {
        const T i = thrust::get<0>(tuple);

        if (data[i] == thrust::get<1>(tuple)) {
            data[i] = thrust::get<2>(tuple);
        }
    }

    thrust::device_ptr<T> data;
};

template <class T>
static inline void
Replace(T *const          data,
        const std::size_t data_size,
        const T *const    to_replace,
        const T *const    values,
        const std::size_t replacement_size) {
    const thrust::device_ptr<const T> data_begin(data);
    const thrust::device_ptr<const T> data_end =
      data_begin + static_cast<std::ptrdiff_t>(data_size);

    const thrust::device_ptr<const T> from_begin(to_replace);
    const thrust::device_ptr<const T> from_end =
      from_begin + static_cast<std::ptrdiff_t>(replacement_size);

    const thrust::device_ptr<const T> to_begin(values);
    const thrust::device_ptr<const T> to_end =
      to_begin + static_cast<std::ptrdiff_t>(replacement_size);

    thrust::device_vector<T> lower_bounds(replacement_size);
    thrust::lower_bound(
      data_begin, data_end, from_begin, from_end, lower_bounds.begin());

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                       lower_bounds.cbegin(), from_begin, to_begin)),
                     thrust::make_zip_iterator(thrust::make_tuple(
                       lower_bounds.cend(), from_end, to_end)),
                     ReplaceFunctor<T>(data));
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
    if (NotEqualReplacementSize(to_replace, values)) { return GDF_CUDA_ERROR; }

    if (NotSameDType(column, to_replace, values)) { return GDF_CUDA_ERROR; }

    switch (column->dtype) {
#define WHEN(DTYPE)                                                           \
    case GDF_##DTYPE: {                                                       \
        using value_type = gdf_dtype_traits<GDF_##DTYPE>::value_type;         \
        Replace<value_type>(static_cast<value_type *>(column->data),          \
                            column->size,                                     \
                            static_cast<value_type *>(to_replace->data),      \
                            static_cast<value_type *>(values->data),          \
                            values->size);                                    \
    } break

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
    default: return GDF_CUDA_ERROR;
    }

    return GDF_SUCCESS;
}
