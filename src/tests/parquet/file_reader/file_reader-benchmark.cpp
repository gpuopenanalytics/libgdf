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

#include <benchmark/benchmark.h>

#include "column_reader.h"
#include "file_reader.h"

#ifndef PARQUET_FILE_PATH
#error PARQUET_FILE_PATH must be defined for precompiling
#define PARQUET_FILE_PATH "/"
#endif

enum ReaderType : std::uint8_t { kGdf };

template <ReaderType T>
struct Readers {};

#define READER_FACTORY(PREFIX)                                                \
    template <>                                                               \
    struct Readers<kGdf> {                                                    \
        typedef typename PREFIX::parquet::BoolReader   BoolReader;            \
        typedef typename PREFIX::parquet::Int64Reader  Int64Reader;           \
        typedef typename PREFIX::parquet::DoubleReader DoubleReader;          \
        typedef typename PREFIX::parquet::FileReader   FileReader;            \
    }

READER_FACTORY(gdf);

template <ReaderType T>
inline static void
readRowGroup(const std::unique_ptr<typename Readers<T>::FileReader> &reader) {
    const std::shared_ptr<::parquet::RowGroupReader> row_group =
      reader->RowGroup(0);
    constexpr static int BATCH_SIZE = 50000;
    std::int16_t definition_level[BATCH_SIZE];
    std::int16_t repetition_level[BATCH_SIZE];
    std::uint8_t valid_bits[BATCH_SIZE];
    std::int64_t levels_read;
    std::int64_t values_read = 0;
    std::int64_t nulls_count;

    std::shared_ptr<parquet::ColumnReader> column;

    column = row_group->Column(0);
    typename Readers<T>::BoolReader *bool_reader =
      static_cast<typename Readers<T>::BoolReader *>(column.get());
    while (bool_reader->HasNext()) {
        bool value[BATCH_SIZE];
        bool_reader->ReadBatchSpaced(BATCH_SIZE,
                                     definition_level,
                                     repetition_level,
                                     value,
                                     valid_bits,
                                     0,
                                     &levels_read,
                                     &values_read,
                                     &nulls_count);
    }

    column = row_group->Column(1);
    typename Readers<T>::Int64Reader *int64_reader =
      static_cast<typename Readers<T>::Int64Reader *>(column.get());
    while (int64_reader->HasNext()) {
        std::int64_t value[BATCH_SIZE];
        int64_reader->ReadBatchSpaced(BATCH_SIZE,
                                      definition_level,
                                      repetition_level,
                                      value,
                                      valid_bits,
                                      0,
                                      &levels_read,
                                      &values_read,
                                      &nulls_count);
    }

    column = row_group->Column(2);
    typename Readers<T>::DoubleReader *double_reader =
      static_cast<typename Readers<T>::DoubleReader *>(column.get());
    while (double_reader->HasNext()) {
        double value[BATCH_SIZE];
        double_reader->ReadBatchSpaced(BATCH_SIZE,
                                       definition_level,
                                       repetition_level,
                                       value,
                                       valid_bits,
                                       0,
                                       &levels_read,
                                       &values_read,
                                       &nulls_count);
    }
}

template <ReaderType T>
static void
BM_FileRead(benchmark::State &state) {
    #ifdef GDF_DECODER_GPU_VERSION
        std::cout << "use gpu decoder\n";
    #else
        std::cout << "use cpu decoder\n";
    #endif
    for (auto _ : state) {
        std::unique_ptr<typename Readers<T>::FileReader> reader =
          Readers<T>::FileReader::OpenFile(PARQUET_FILE_PATH);
        readRowGroup<T>(reader);
    }
}
BENCHMARK_TEMPLATE(BM_FileRead, kGdf);
