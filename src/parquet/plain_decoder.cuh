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

#include <arrow/util/bit-stream-utils.h>
#include "../arrow/bit-stream.h"
#include <thrust/device_vector.h>

namespace parquet {
class ColumnDescriptor;
}

namespace gdf {
namespace parquet {
namespace internal {

template <typename DataType>
class PlainDecoder : public ::parquet::Decoder<DataType> {
public:
    typedef typename DataType::c_type T;
    using ::parquet::Decoder<DataType>::num_values_;

    explicit PlainDecoder(const ::parquet::ColumnDescriptor *descr)
      : ::parquet::Decoder<DataType>(descr, ::parquet::Encoding::PLAIN),
        data_(nullptr), len_(0) {
        if (descr_
            && descr_->physical_type()
                 == ::parquet::Type::FIXED_LEN_BYTE_ARRAY) {
            type_length_ = descr_->type_length();
        } else {
            type_length_ = -1;
        }
    }

    virtual void
    SetData(int num_values, const std::uint8_t *data, int len) {
        num_values_ = num_values;
        data_       = data;
        len_        = len;
    }

    virtual int Decode(T *buffer, int max_values);

private:
    using ::parquet::Decoder<DataType>::descr_;
    const std::uint8_t *data_;
    int                 len_;
    int                 type_length_;
};

template <typename T>
inline int
DecodePlain(const std::uint8_t *data,
            std::int64_t        data_size,
            int                 num_values,
            int,
            T *out) {
    int bytes_to_decode = num_values * static_cast<int>(sizeof(T));
    if (data_size < bytes_to_decode) {
        ::parquet::ParquetException::EofException();
    }
    // std::memcpy(out, data, bytes_to_decode);
    cudaMemcpy(out, data, bytes_to_decode, cudaMemcpyHostToDevice);
    return bytes_to_decode;
}

// template <>
// inline int
// DecodePlain<::parquet::ByteArray>(const std::uint8_t *data,
//                                   std::int64_t        data_size,
//                                   int                 num_values,
//                                   int,
//                                   ::parquet::ByteArray *out) {
//     int bytes_decoded = 0;
//     int increment;
//     for (int i = 0; i < num_values; ++i) {
//         std::uint32_t len = out[i].len =
//           *reinterpret_cast<const std::uint32_t *>(data);
//         increment = static_cast<int>(sizeof(std::uint32_t) + len);
//         if (data_size < increment) ::parquet::ParquetException::EofException();
//         out[i].ptr = data + sizeof(std::uint32_t);
//         data += increment;
//         data_size -= increment;
//         bytes_decoded += increment;
//     }
//     return bytes_decoded;
// }

// template <>
// inline int
// DecodePlain<::parquet::FixedLenByteArray>(const std::uint8_t *data,
//                                           std::int64_t        data_size,
//                                           int                 num_values,
//                                           int                 type_length,
//                                           ::parquet::FixedLenByteArray *out) {
//     int bytes_to_decode = type_length * num_values;
//     if (data_size < bytes_to_decode) {
//         ::parquet::ParquetException::EofException();
//     }
//     for (int i = 0; i < num_values; ++i) {
//         out[i].ptr = data;
//         data += type_length;
//         data_size -= type_length;
//     }
//     return bytes_to_decode;
// }

template <typename DataType>
inline int
PlainDecoder<DataType>::Decode(T *buffer, int max_values) {
    max_values = std::min(max_values, num_values_);
    int bytes_consumed =
      DecodePlain<T>(data_, len_, max_values, type_length_, buffer);
    data_ += bytes_consumed;
    len_ -= bytes_consumed;
    num_values_ -= max_values;
    return max_values;
}

template <>
class PlainDecoder<::parquet::BooleanType>
  : public ::parquet::Decoder<::parquet::BooleanType> {
public:
    explicit PlainDecoder(const ::parquet::ColumnDescriptor *descr)
      : ::parquet::Decoder<::parquet::BooleanType>(
          descr,
          ::parquet::Encoding::PLAIN) {}

    virtual void
    SetData(int num_values, const std::uint8_t *data, int len) {
        num_values_ = num_values;
        bit_reader_ = gdf::arrow::internal::BitReader(data, len);
    }

    int
    Decode(std::uint8_t *buffer, int max_values) {
        max_values = std::min(max_values, num_values_);
        bool val;
        for (int i = 0; i < max_values; ++i) {
            if (!bit_reader_.GetValue(1, &val)) {
                ::parquet::ParquetException::EofException();
            }
            ::arrow::BitUtil::SetArrayBit(buffer, i, val);
        }
        num_values_ -= max_values;
        return max_values;
    }

    virtual int
    Decode(bool *buffer, int max_values) {
        max_values = std::min(max_values, num_values_);

        int literal_batch = max_values;
        int values_read = 0;
        std::vector<uint32_t> rleRuns;
        std::vector<uint64_t> rleValues;
        int numRle;
        int numBitpacked;
        std::vector<int> unpack32InputOffsets, unpack32InputRunLengths, unpack32OutputOffsets;
        std::vector<int> remainderInputOffsets, remainderBitOffsets, remainderSetSize,
                remainderOutputOffsets;

        bit_reader_.SetGpuBatchMetadata(
                1, buffer, literal_batch, values_read, unpack32InputOffsets, unpack32InputRunLengths,
                unpack32OutputOffsets, remainderInputOffsets, remainderBitOffsets,
                remainderSetSize, remainderOutputOffsets);

        gdf::arrow::internal::unpack_using_gpu<bool> (
                bit_reader_.get_buffer(), bit_reader_.get_buffer_len(),
                unpack32InputOffsets,
				unpack32InputRunLengths,
                unpack32OutputOffsets,
                remainderInputOffsets, remainderBitOffsets, remainderSetSize,
                remainderOutputOffsets, 1, buffer, literal_batch);

        // if (bit_reader_.GetBatch(1, buffer, max_values) != max_values) {
        //     ::parquet::ParquetException::EofException();
        // }
        num_values_ -= max_values;
        return max_values;
    }

private:
    gdf::arrow::internal::BitReader bit_reader_;
};

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
