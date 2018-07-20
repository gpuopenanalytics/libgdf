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

#ifndef _GDF_PARQUET_COLUMN_READER_H
#define _GDF_PARQUET_COLUMN_READER_H

#include <parquet/column_reader.h>

namespace gdf {
namespace parquet {

template <class DataType>
class ColumnReader : public ::parquet::ColumnReader {
public:
    typedef typename DataType::c_type T;

    bool HasNext();

    std::int64_t ReadBatch(std::int64_t  batchSize,
                           std::int16_t *definitionLevels,
                           std::int16_t *repetitionLevels,
                           T *           values,
                           std::int64_t *valuesRead);

private:
    bool ReadNewPage() final;

    typedef ::parquet::Decoder<DataType>                  DecoderType;
    std::unordered_map<int, std::shared_ptr<DecoderType>> decoders_;
    DecoderType *                                         current_decoder_;
};

using BoolReader              = ColumnReader<::parquet::BooleanType>;
using Int32Reader             = ColumnReader<::parquet::Int32Type>;
using Int64Reader             = ColumnReader<::parquet::Int64Type>;
using Int96Reader             = ColumnReader<::parquet::Int96Type>;
using FloatReader             = ColumnReader<::parquet::FloatType>;
using DoubleReader            = ColumnReader<::parquet::DoubleType>;
using ByteArrayReader         = ColumnReader<::parquet::ByteArrayType>;
using FixedLenByteArrayReader = ColumnReader<::parquet::FLBAType>;

}  // namespace parquet
}  // namespace gdf

#endif
