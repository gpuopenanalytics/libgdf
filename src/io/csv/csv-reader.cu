/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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


#include "gdf/gdf.h"
#include "gdf/errorutils.h"


gdf_error read_csv(csv_read_arg *args) {

	int num_cols = 2;
	int num_rows = 100;


	gdf_column **cols = (gdf_column **)malloc( sizeof(gdf_column *) * num_cols);


	for ( int x = 0; x < num_cols; x++) {
		gdf_column *gdf = (gdf_column *)malloc(sizeof(gdf_column) * 1);

		gdf->size		= num_rows;
		gdf->dtype		= GDF_INT64;
		//gdf->null_count	= 0;

		CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int64_t) * num_rows)));

		cols[x] = gdf;
	}


	args->num_cols 	= num_cols;
	args->num_rows 	= num_rows;
	args->data 		= cols;


	return GDF_SUCCESS;
}
