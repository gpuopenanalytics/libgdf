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

#include <mutex>
#include <thread>
#include <numeric>

#include <arrow/util/bit-util.h>
#include <arrow/util/logging.h>

#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include <thrust/device_ptr.h>

#include "column_reader.h"
#include "file_reader.h"

#include <gdf/parquet/api.h>

BEGIN_NAMESPACE_GDF_PARQUET

namespace {

struct ParquetTypeHash {
    template <class T>
    std::size_t
    operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

const std::unordered_map<::parquet::Type::type, gdf_dtype, ParquetTypeHash>
  dtype_from_physical_type_map{
    {::parquet::Type::BOOLEAN, GDF_INT8},
    {::parquet::Type::INT32, GDF_INT32},
    {::parquet::Type::INT64, GDF_INT64},
    {::parquet::Type::FLOAT, GDF_FLOAT32},
    {::parquet::Type::DOUBLE, GDF_FLOAT64},
  };

const std::
  unordered_map<::parquet::LogicalType::type, gdf_dtype, ParquetTypeHash>
    dtype_from_logical_type_map{
      {::parquet::LogicalType::NONE, GDF_invalid},
      {::parquet::LogicalType::UTF8, GDF_invalid},
      {::parquet::LogicalType::MAP, GDF_invalid},
      {::parquet::LogicalType::MAP_KEY_VALUE, GDF_invalid},
      {::parquet::LogicalType::LIST, GDF_invalid},
      {::parquet::LogicalType::ENUM, GDF_invalid},
      {::parquet::LogicalType::DECIMAL, GDF_invalid},
      {::parquet::LogicalType::DATE, GDF_DATE32},
      {::parquet::LogicalType::TIME_MILLIS, GDF_invalid},
      {::parquet::LogicalType::TIME_MICROS, GDF_invalid},
      {::parquet::LogicalType::TIMESTAMP_MILLIS, GDF_TIMESTAMP},
      {::parquet::LogicalType::TIMESTAMP_MICROS, GDF_invalid},
      {::parquet::LogicalType::UINT_8, GDF_invalid},
      {::parquet::LogicalType::UINT_16, GDF_invalid},
      {::parquet::LogicalType::UINT_32, GDF_invalid},
      {::parquet::LogicalType::UINT_64, GDF_invalid},
      {::parquet::LogicalType::INT_8, GDF_INT8},
      {::parquet::LogicalType::INT_16, GDF_INT16},
      {::parquet::LogicalType::INT_32, GDF_INT32},
      {::parquet::LogicalType::INT_64, GDF_INT64},
      {::parquet::LogicalType::JSON, GDF_invalid},
      {::parquet::LogicalType::BSON, GDF_invalid},
      {::parquet::LogicalType::INTERVAL, GDF_invalid},
      {::parquet::LogicalType::NA, GDF_invalid},
    };

static inline gdf_dtype
_DTypeFrom(const ::parquet::ColumnDescriptor *const column_descriptor) {
    const ::parquet::LogicalType::type logical_type =
      column_descriptor->logical_type();

    if (logical_type != ::parquet::LogicalType::NONE) {
        return dtype_from_logical_type_map.at(logical_type);
    }

    const ::parquet::Type::type physical_type =
      column_descriptor->physical_type();

    return dtype_from_physical_type_map.at(physical_type);
}

static inline gdf_error
_ReadColumn(const std::shared_ptr<GdfRowGroupReader> &row_group_reader,
            const std::vector<std::size_t> &          column_indices,
            std::size_t                               offsets[],
            gdf_column *const                         gdf_columns) {
    for (std::size_t column_reader_index = 0;
         column_reader_index < column_indices.size();
         column_reader_index++) {
        const gdf_column &_gdf_column = gdf_columns[column_reader_index];
        const std::shared_ptr<::parquet::ColumnReader> column_reader =
          row_group_reader->Column(
            static_cast<int>(column_indices[column_reader_index]));

        switch (column_reader->type()) {
#define WHEN(TYPE)                                                            \
    case ::parquet::Type::TYPE: {                                             \
        std::shared_ptr<gdf::parquet::ColumnReader<                           \
          ::parquet::DataType<::parquet::Type::TYPE>>>                        \
          reader = std::static_pointer_cast<gdf::parquet::ColumnReader<       \
            ::parquet::DataType<::parquet::Type::TYPE>>>(column_reader);      \
        if (reader->HasNext()) {                                              \
            offsets[column_reader_index] +=                                   \
              reader->ToGdfColumn(_gdf_column, offsets[column_reader_index]); \
        }                                                                     \
    } break
            WHEN(BOOLEAN);
            WHEN(INT32);
            WHEN(INT64);
            WHEN(FLOAT);
            WHEN(DOUBLE);
        default:
#ifdef GDF_DEBUG
            std::cerr << "Column type error from file" << std::endl;
#endif
            return GDF_IO_ERROR;  //TODO: improve using exception handling
#undef WHEN
        }
    }
    return GDF_SUCCESS;
}

static inline gdf_error
_ReadFile(const std::unique_ptr<FileReader> &file_reader,
          const std::vector<std::size_t> &   indices,
          gdf_column *const                  gdf_columns) {
    const std::shared_ptr<::parquet::FileMetaData> &metadata =
      file_reader->metadata();
    const std::size_t num_rows =
      static_cast<std::size_t>(metadata->num_rows());
    const std::size_t num_row_groups =
      static_cast<std::size_t>(metadata->num_row_groups());

    std::size_t offsets[indices.size()];
    for (std::size_t i = 0; i < indices.size(); i++) { offsets[i] = 0; }

    for (std::size_t row_group_index = 0; row_group_index < num_row_groups;
         row_group_index++) {
        const auto row_group_reader =
          file_reader->RowGroup(static_cast<int>(row_group_index));

        gdf_error status =
          _ReadColumn(row_group_reader, indices, offsets, gdf_columns);
        if (status != GDF_SUCCESS) { return status; }
    }

    return GDF_SUCCESS;
}

static inline gdf_error
_ReadFile(const std::unique_ptr<FileReader> &file_reader,
          const std::vector<std::size_t> &   row_group_indices,
          const std::vector<std::size_t> &   column_indices,
          gdf_column *const                  gdf_columns) {
    const std::shared_ptr<::parquet::FileMetaData> &metadata =
      file_reader->metadata();
    const std::size_t num_rows =
      static_cast<std::size_t>(metadata->num_rows());

    std::size_t offsets[column_indices.size()];
    for (std::size_t i = 0; i < column_indices.size(); i++) { offsets[i] = 0; }

    for (const std::size_t row_group_index : row_group_indices) {
        const auto row_group_reader =
          file_reader->RowGroup(static_cast<int>(row_group_index));

        gdf_error status =
          _ReadColumn(row_group_reader, column_indices, offsets, gdf_columns);
        if (status != GDF_SUCCESS) { return status; }
    }

    return GDF_SUCCESS;
}


struct ParquetReaderJob {

	std::size_t row_group_index;
	std::size_t column_index;
	std::size_t column_index_in_read_set;

//	std::shared_ptr<GdfRowGroupReader> row_group_reader;
	std::shared_ptr<::parquet::ColumnReader> column_reader;

	const gdf_column & column;
	std::size_t offset;

	gdf_valid_type first_valid_byte;
	gdf_valid_type last_valid_byte;

	ParquetReaderJob(std::size_t _row_group_index,
	std::size_t _column_index,
	std::size_t _column_index_in_read_set,
//	std::shared_ptr<GdfRowGroupReader> _row_group_reader,
	std::shared_ptr<::parquet::ColumnReader> _column_reader,
	const gdf_column & _column,
	std::size_t _offset )
	: row_group_index(_row_group_index),
	  column_index(_column_index),
	  column_index_in_read_set(_column_index_in_read_set),
//	  row_group_reader(std::move(_row_group_reader)),
	  column_reader(std::move(_column_reader)),
	  column(std::move(_column)),
	  offset(_offset)
	{}
};



void _ProcessParquetReaderJobsThread(std::vector<ParquetReaderJob> & jobs, std::mutex & lock,
		int & job_index, gdf_error & gdf_error_out){

	lock.lock();
	int current_job = job_index;
	job_index++;
	lock.unlock();

	gdf_error current_gdf_error = GDF_SUCCESS;

	while (current_job < jobs.size()){

		switch (jobs[current_job].column_reader->type()) {
		#define WHEN(TYPE)                                                            \
				case ::parquet::Type::TYPE: {                                             \
					std::shared_ptr<gdf::parquet::ColumnReader<                           \
					::parquet::DataType<::parquet::Type::TYPE>>>                        \
					 reader = std::static_pointer_cast<gdf::parquet::ColumnReader<       \
					 ::parquet::DataType<::parquet::Type::TYPE>>>(jobs[current_job].column_reader);      \
					 if (reader->HasNext()) {                                              \
						 reader->ToGdfColumn(jobs[current_job].column, jobs[current_job].offset, jobs[current_job].first_valid_byte, jobs[current_job].last_valid_byte); \
					 }                                                                     \
				} break
				WHEN(BOOLEAN);
				WHEN(INT32);
				WHEN(INT64);
				WHEN(FLOAT);
				WHEN(DOUBLE);
				default:
		#ifdef GDF_DEBUG
					std::cerr << "Column type error from file" << std::endl;
		#endif
					current_gdf_error =  GDF_IO_ERROR;  //TODO: improve using exception handling
		#undef WHEN
				}


		lock.lock();
		if (gdf_error_out != GDF_SUCCESS){ // if error we want to exit
			current_job = jobs.size();
		} else if (current_gdf_error != GDF_SUCCESS) { // if error we want to exit
			gdf_error_out = current_gdf_error;
			current_job = jobs.size();
		} else {
			current_job = job_index;
		}
		job_index++;
		lock.unlock();
	}

}

gdf_error _ProcessParquetReaderJobs(std::vector<ParquetReaderJob> & jobs){

	std::mutex lock;
	int job_index = 0;
	gdf_error gdf_error_out = GDF_SUCCESS;

	int num_threads = std::thread::hardware_concurrency();
	num_threads = jobs.size() < num_threads ? jobs.size() : num_threads;


//	_ProcessParquetReaderJobsThread(jobs, lock, job_index, gdf_error_out);

	std::vector<std::thread> threads(num_threads);

	for (int i = 0; i < num_threads; i++){
		threads[i] = std::thread(_ProcessParquetReaderJobsThread,
				std::ref(jobs), std::ref(lock), std::ref(job_index), std::ref(gdf_error_out));
	}
	for (int i = 0; i < num_threads; i++){
		threads[i].join();
	}



	return gdf_error_out;
}


static inline gdf_error
_ReadFileMultiThread(const std::unique_ptr<FileReader> &file_reader,
		const std::vector<std::size_t> &   row_group_indices,
		const std::vector<std::size_t> &   column_indices,
		gdf_column *const                  gdf_columns) {
	const std::shared_ptr<::parquet::FileMetaData> &metadata =
			file_reader->metadata();
	const std::size_t num_rows =
			static_cast<std::size_t>(metadata->num_rows());


	std::vector<ParquetReaderJob> jobs;

	std::vector<std::size_t> offsets(row_group_indices.size(), 0);

	for (std::size_t row_group_index_in_set = 0; row_group_index_in_set < row_group_indices.size();
			row_group_index_in_set++) {

		std::size_t row_group_index = row_group_indices[row_group_index_in_set];

		const auto row_group_reader =
				file_reader->RowGroup(static_cast<int>(row_group_index));

		int64_t num_rows = row_group_reader->metadata()->num_rows();


		for (std::size_t column_reader_index = 0;
				column_reader_index < column_indices.size();
				column_reader_index++) {
			const gdf_column &_gdf_column = gdf_columns[column_reader_index];
			const std::shared_ptr<::parquet::ColumnReader> column_reader =
					row_group_reader->Column(
							static_cast<int>(column_indices[column_reader_index]));

			jobs.emplace_back(row_group_index, column_indices[column_reader_index],
					column_reader_index, column_reader,
					_gdf_column, offsets[row_group_index_in_set]);

		}

		if (row_group_index_in_set < row_group_indices.size() - 1){
			offsets[row_group_index_in_set + 1] = offsets[row_group_index_in_set] + num_rows;
		}
	}

	gdf_error gdf_error_out = _ProcessParquetReaderJobs(jobs);

	// now lets fix all the valid bytes that were shared for a column accross rowgroups
	if (row_group_indices.size() > 1){
		for (std::size_t column_reader_index = 0; column_reader_index < column_indices.size();
				column_reader_index++) {

			for (std::size_t row_group_index_in_set = 0; row_group_index_in_set < row_group_indices.size() - 1;
					row_group_index_in_set++) {

				int job_index1 = (row_group_index_in_set * column_indices.size()) + column_reader_index;
				int job_index2 = ((row_group_index_in_set + 1) * column_indices.size()) + column_reader_index;

				gdf_valid_type merged = jobs[job_index1].last_valid_byte | jobs[job_index2].first_valid_byte;

				// determine location of where the merged byte goes
				// copy merged into valid
				std::size_t merged_byte_offset = (offsets[row_group_index_in_set + 1]/8);

				cudaMemcpy(gdf_columns[column_reader_index].valid + merged_byte_offset, &merged, sizeof(gdf_valid_type), cudaMemcpyHostToDevice);
			}
		}
	}


	return gdf_error_out;
}



template <::parquet::Type::type TYPE>
static inline gdf_error
_AllocateGdfColumn(const std::size_t                        num_rows,
                   const ::parquet::ColumnDescriptor *const column_descriptor,
                   gdf_column &                             _gdf_column) {
    const std::size_t value_byte_size =
      static_cast<std::size_t>(::parquet::type_traits<TYPE>::value_byte_size);

    cudaError_t status =
      cudaMalloc(&_gdf_column.data, num_rows * value_byte_size);
    if (status != cudaSuccess) {
#ifdef GDF_DEBUG
        std::cerr << "Allocation error for data\n" << e.what() << std::endl;
#endif
        return GDF_IO_ERROR;
    }

    status = cudaMalloc(reinterpret_cast<void **>(&_gdf_column.valid),
                        ::arrow::BitUtil::BytesForBits(num_rows));
    if (status != cudaSuccess) {
#ifdef GDF_DEBUG
        std::cerr << "Allocation error for valid\n" << e.what() << std::endl;
#endif
        return GDF_IO_ERROR;
    }

    _gdf_column.size  = num_rows;
    _gdf_column.dtype = _DTypeFrom(column_descriptor);

    return GDF_SUCCESS;
}  // namespace

static inline std::vector<const ::parquet::ColumnDescriptor *>
_ColumnDescriptorsFrom(const std::unique_ptr<FileReader> &file_reader,
                       const std::vector<std::size_t> &   indices) {
    const auto &row_group_reader = file_reader->RowGroup(0);

    std::vector<const ::parquet::ColumnDescriptor *> column_descriptors;
    column_descriptors.reserve(indices.size());

    for (const std::size_t i : indices) {
        column_descriptors.emplace_back(row_group_reader->Column(i)->descr());
    }

    return column_descriptors;
}

static inline gdf_error
_AllocateGdfColumns(const std::unique_ptr<FileReader> &file_reader,
		const std::vector<std::size_t> &   row_group_indices,
		const std::vector<std::size_t> &   column_indices,
		gdf_column *const                  gdf_columns) {
    const std::vector<const ::parquet::ColumnDescriptor *> column_descriptors =
      _ColumnDescriptorsFrom(file_reader, column_indices);

    int64_t num_rows = 0;
    for (std::size_t row_group_index_in_set = 0; row_group_index_in_set < row_group_indices.size();
    		row_group_index_in_set++) {

    	std::size_t row_group_index = row_group_indices[row_group_index_in_set];

    	const auto row_group_reader =
    			file_reader->RowGroup(static_cast<int>(row_group_index));

    	num_rows += row_group_reader->metadata()->num_rows();
    }


    const std::size_t num_columns = column_indices.size();


#define WHEN(TYPE)                                                            \
    case ::parquet::Type::TYPE:                                               \
        _AllocateGdfColumn<::parquet::Type::TYPE>(                            \
          num_rows, column_descriptor, _gdf_column);                          \
        break

    for (std::size_t i = 0; i < num_columns; i++) {
        gdf_column &                             _gdf_column = gdf_columns[i];
        const ::parquet::ColumnDescriptor *const column_descriptor =
          column_descriptors[i];

        switch (column_descriptor->physical_type()) {
            WHEN(BOOLEAN);
            WHEN(INT32);
            WHEN(INT64);
            WHEN(FLOAT);
            WHEN(DOUBLE);
        default:
#ifdef GDF_DEBUG
            std::cerr << "Column type not supported" << std::endl;
#endif
            return GDF_IO_ERROR;
        }
    }
#undef WHEN
    return GDF_SUCCESS;
}

static inline gdf_column *
_CreateGdfColumns(const std::size_t num_columns) try {
    return new gdf_column[num_columns];
} catch (const std::bad_alloc &e) {
#ifdef GDF_DEBUG
    std::cerr << "Allocation error for gdf columns\n" << e.what() << std::endl;
#endif
    return nullptr;
}


static inline std::vector<std::size_t>
_GetColumnIndices(const std::unique_ptr<FileReader> &file_reader,
		const char *const *const raw_names){

	 std::vector<std::size_t> indices;

	const std::shared_ptr<const ::parquet::FileMetaData> &metadata =
			file_reader->metadata();

	const std::size_t num_columns =
			static_cast<std::size_t>(metadata->num_columns());

	 auto schema = file_reader->RowGroup(0)->metadata()->schema();

	 std::vector<std::pair<std::string, std::size_t>> parquet_columns;
	 parquet_columns.reserve(num_columns);

	 for (std::size_t i = 0; i < num_columns; i++) {
		 if (schema->Column(i)->physical_type() != ::parquet::Type::BYTE_ARRAY &&
				 schema->Column(i)->physical_type() != ::parquet::Type::FIXED_LEN_BYTE_ARRAY){

			 parquet_columns.push_back(std::make_pair(schema->Column(i)->name(), i));

			 std::cout<<"parquet_columns: "<<parquet_columns[parquet_columns.size() -1].first<<"  "<<parquet_columns[parquet_columns.size() -1].second<<std::endl;
		 } else {
			 std::cout<<"skipped one: "<<schema->Column(i)->name()<<std::endl;
		 }

	 }

	if (raw_names != nullptr) {
		for (const char *const *name_ptr = raw_names; *name_ptr != nullptr;
				name_ptr++) {

			std::string filter_name = *name_ptr;
			for (std::size_t i = 0; i < parquet_columns.size(); i++) {
				if (filter_name == parquet_columns[i].first){
					indices.push_back(parquet_columns[i].second);
					break;
				}
			}
		}
	} else {
		for (std::size_t i = 0; i < parquet_columns.size(); i++) {
			indices.push_back(parquet_columns[i].second);
		}
	}
	return indices;
}


static inline gdf_error
_CheckMinimalData(const std::unique_ptr<FileReader> &file_reader) {
    const std::shared_ptr<const ::parquet::FileMetaData> &metadata =
      file_reader->metadata();

    if (metadata->num_row_groups() == 0) { return GDF_IO_ERROR; }

    if (metadata->num_rows() == 0) { return GDF_IO_ERROR; }

    return GDF_SUCCESS;
}

static inline std::unique_ptr<FileReader>
_OpenFile(const std::string &filename) try {
    return FileReader::OpenFile(filename);
} catch (std::exception &e) {
#ifdef GDF_DEBUG
    std::cerr << "Open file\n" << e.what() << std::endl;
#endif
    return nullptr;
}

}  // namespace

gdf_error
read_parquet_by_ids(const std::string &             filename,
                    const std::vector<std::size_t> &row_group_indices,
                    const std::vector<std::size_t> &column_indices,
                    std::vector<gdf_column *> &     out_gdf_columns) {
    const std::unique_ptr<FileReader> file_reader = _OpenFile(filename);

    if (!file_reader) { return GDF_IO_ERROR; }

    if (_CheckMinimalData(file_reader) != GDF_SUCCESS) { return GDF_IO_ERROR; }

    gdf_column *const gdf_columns = _CreateGdfColumns(column_indices.size());

    if (gdf_columns == nullptr) { return GDF_IO_ERROR; }

    if (_AllocateGdfColumns(file_reader, row_group_indices, column_indices, gdf_columns)
        != GDF_SUCCESS) {
        return GDF_IO_ERROR;
    }

    if (_ReadFileMultiThread(file_reader, row_group_indices, column_indices, gdf_columns)
        != GDF_SUCCESS) {
        return GDF_IO_ERROR;
    }

    for (std::size_t i = 0; i < column_indices.size(); i++) {
        out_gdf_columns.push_back(&gdf_columns[i]);
    }

    return GDF_SUCCESS;
}

extern "C" {

gdf_error
read_parquet(const char *const        filename,
             const char *const *const columns,
             gdf_column **const       out_gdf_columns,
             size_t *const            out_gdf_columns_length) {
    const std::unique_ptr<FileReader> file_reader = _OpenFile(filename);

    if (!file_reader) { return GDF_IO_ERROR; }

    if (_CheckMinimalData(file_reader) != GDF_SUCCESS) { return GDF_IO_ERROR; }

    const std::vector<std::size_t> column_indices =
    		_GetColumnIndices(file_reader, columns);

    gdf_column *const gdf_columns = _CreateGdfColumns(column_indices.size());

    if (gdf_columns == nullptr) { return GDF_IO_ERROR; }

    const std::shared_ptr<::parquet::FileMetaData> &metadata =
    		file_reader->metadata();
    const std::size_t num_row_groups =
    		static_cast<std::size_t>(metadata->num_row_groups());

    std::vector<std::size_t> row_group_indices(num_row_groups);
    std::iota( row_group_indices.begin(), row_group_indices.end(), 0);

    if (_AllocateGdfColumns(file_reader, row_group_indices, column_indices, gdf_columns)
        != GDF_SUCCESS) {
        return GDF_IO_ERROR;
    }
    if (_ReadFileMultiThread(file_reader, row_group_indices, column_indices, gdf_columns) != GDF_SUCCESS) {
        return GDF_IO_ERROR;
    }

    *out_gdf_columns        = gdf_columns;
    *out_gdf_columns_length = column_indices.size();

    return GDF_SUCCESS;
}
}

END_NAMESPACE_GDF_PARQUET
