#include <benchmark/benchmark.h>

#include <gdf/parquet/api.h>

#include "column_reader.h"
#include "file_reader.h"

#include "../../helper/utils.cuh"

#ifndef PARQUET_FILE_PATH
#error PARQUET_FILE_PATH must be defined for precompiling
#define PARQUET_FILE_PATH "/"
#endif

static void
BM_FileRead_mt(benchmark::State &state) {

	for (auto _ : state) {

		gdf_column *columns = nullptr;
		std::size_t columns_length = 0;
		gdf_error error_code = gdf::parquet::read_parquet(
				PARQUET_FILE_PATH, nullptr, &columns, &columns_length);

	}
}


// NOTE: this way of doing the reading singlethreaded adds some overhead.
static void
BM_FileRead_st(benchmark::State &state) {

	for (auto _ : state) {

		const std::unique_ptr<gdf::parquet::FileReader> file_reader = gdf::parquet::FileReader::OpenFile(PARQUET_FILE_PATH);

		std::shared_ptr<parquet::FileMetaData> file_metadata = file_reader->metadata();
		const parquet::SchemaDescriptor *schema = file_metadata->schema();

		int numRowGroups = file_metadata->num_row_groups();
		int num_columns = file_metadata->num_columns();

		auto row_group_reader = file_reader->RowGroup(0);

		for (int rg = 0; rg < numRowGroups; rg++){
			for (int col = 0; col < num_columns; col++){

				if (row_group_reader->Column(col)->descr()->physical_type() != ::parquet::Type::BYTE_ARRAY &&
						row_group_reader->Column(col)->descr()->physical_type() != ::parquet::Type::FIXED_LEN_BYTE_ARRAY){

					std::vector<std::size_t> row_group_indices(1, rg);
					std::vector<std::size_t> column_indices(1, col);

					std::vector<gdf_column *>  out_gdf_columns;
					gdf_error error_code = gdf::parquet::read_parquet_by_ids(
							PARQUET_FILE_PATH, row_group_indices, column_indices, out_gdf_columns);

				}
			}
		}
	}
}

BENCHMARK(BM_FileRead_mt);
BENCHMARK(BM_FileRead_st);
