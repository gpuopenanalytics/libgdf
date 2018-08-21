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

#include "../arrow/rle_decoder.h"
#include "arrow/util/rle-encoding.h"

namespace parquet {
class ColumnDescriptor;
}

namespace gdf {
namespace parquet {
namespace internal {

template <typename Type, typename RleDecoder>
class DictionaryDecoder : public ::parquet::Decoder<Type> {
public:
    typedef typename Type::c_type T;

    explicit DictionaryDecoder(
      const ::parquet::ColumnDescriptor *descr,
      ::arrow::MemoryPool *              pool = ::arrow::default_memory_pool())
      : ::parquet::Decoder<Type>(descr, ::parquet::Encoding::RLE_DICTIONARY),
        dictionary_(0, pool),
        byte_array_data_(::parquet::AllocateBuffer(pool, 0)) {}

    void SetDict(::parquet::Decoder<Type> *dictionary);

    void
    SetData(int num_values, const std::uint8_t *data, int len) override {
        num_values_ = num_values;
        if (len == 0) return;
        std::uint8_t bit_width = *data;
        ++data;
        --len;
        idx_decoder_ = RleDecoder(data, len, bit_width);
    }

    int
    Decode(T *buffer, int max_values) override {
        max_values         = std::min(max_values, num_values_);
        int decoded_values = idx_decoder_.GetBatchWithDict(
          dictionary_.data(), num_dictionary_values_, buffer, max_values);
        if (decoded_values != max_values) {
            ::parquet::ParquetException::EofException();
        }
        num_values_ -= max_values;
        return max_values;
    }

    int
    DecodeSpaced(T *                 buffer,
                 int                 num_values,
                 int                 null_count,
                 const std::uint8_t *valid_bits,
                 std::int64_t        valid_bits_offset) override {
        int decoded_values =
          idx_decoder_.GetBatchWithDictSpaced(dictionary_.data(),
                                              buffer,
                                              num_values,
                                              null_count,
                                              valid_bits,
                                              valid_bits_offset);
        if (decoded_values != num_values) {
            ::parquet::ParquetException::EofException();
        }
        return decoded_values;
    }

private:
    using ::parquet::Decoder<Type>::num_values_;

    ::parquet::Vector<T> dictionary_;

    std::shared_ptr<::parquet::PoolBuffer> byte_array_data_;

    RleDecoder idx_decoder_;

    int num_dictionary_values_;
};

template <typename Type, typename RleDecoder>
inline void
DictionaryDecoder<Type, RleDecoder>::SetDict(
  ::parquet::Decoder<Type> *dictionary) {
    int num_dictionary_values = dictionary->values_left();
    num_dictionary_values_ = num_dictionary_values;
    dictionary_.Resize(num_dictionary_values);
    dictionary->Decode(&dictionary_[0], num_dictionary_values);
}

template <>
inline void
DictionaryDecoder<::parquet::BooleanType, ::arrow::RleDecoder>::SetDict(
  ::parquet::Decoder<::parquet::BooleanType> *) {
    ::parquet::ParquetException::NYI(
      "Dictionary encoding is not implemented for boolean values");
}

template <>
inline void
DictionaryDecoder<::parquet::ByteArrayType, ::arrow::RleDecoder>::SetDict(
  ::parquet::Decoder<::parquet::ByteArrayType> *dictionary) {
    int num_dictionary_values = dictionary->values_left();
    num_dictionary_values_ = num_dictionary_values;
    dictionary_.Resize(num_dictionary_values);
    dictionary->Decode(&dictionary_[0], num_dictionary_values);

    int total_size = 0;
    for (int i = 0; i < num_dictionary_values; ++i) {
        total_size += dictionary_[i].len;
    }
    if (total_size > 0) {
        PARQUET_THROW_NOT_OK(byte_array_data_->Resize(total_size, false));
    }

    int           offset     = 0;
    std::uint8_t *bytes_data = byte_array_data_->mutable_data();
    for (int i = 0; i < num_dictionary_values; ++i) {
        std::memcpy(
          bytes_data + offset, dictionary_[i].ptr, dictionary_[i].len);
        dictionary_[i].ptr = bytes_data + offset;
        offset += dictionary_[i].len;
    }
}

template <>
inline void
DictionaryDecoder<::parquet::FLBAType, ::arrow::RleDecoder>::SetDict(
  ::parquet::Decoder<::parquet::FLBAType> *dictionary) {
    int num_dictionary_values = dictionary->values_left();
    num_dictionary_values_ = num_dictionary_values;
    dictionary_.Resize(num_dictionary_values);
    dictionary->Decode(&dictionary_[0], num_dictionary_values);

    int fixed_len  = descr_->type_length();
    int total_size = num_dictionary_values * fixed_len;

    PARQUET_THROW_NOT_OK(byte_array_data_->Resize(total_size, false));
    std::uint8_t *bytes_data = byte_array_data_->mutable_data();
    for (std::int32_t i = 0, offset = 0; i < num_dictionary_values;
         ++i, offset += fixed_len) {
        std::memcpy(bytes_data + offset, dictionary_[i].ptr, fixed_len);
        dictionary_[i].ptr = bytes_data + offset;
    }
}

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
