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

/*
 * The code  uses the Thrust library
 */

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "type_conversion.cuh"

#include "gdf/gdf.h"
#include "gdf/errorutils.h"

constexpr int32_t HASH_SEED = 33;

using namespace std;

//-- define the structure for raw data handling - for internal use
typedef struct raw_csv_ {
    char *				data;			// on-device: the raw unprocessed CSV data - loaded as a large char * array
    uint64_t *			rec_bits;		// on-device: bitmap indicator of there a record break is located
    uint64_t *			field_bits;		// on-device: bitmap of where a field break is located (delimiter)
    int	*				recPerChunck;	// on-device: Number of records per bitmap chunks
    long * 				offsets;		// on-device: index of record - for multiple records per chunks it is starting index

    char				delimiter;		// host: the delimiter
    long				num_bytes;		// host: the number of bytes in the data
    long				num_bits;		// host: the number of 64-bit bitmaps (different than valid)
	long            	num_records;  	// host: number of records (per column)
	int					num_cols;		// host: number of columns
    vector<gdf_dtype>	dtypes;			// host: array of dtypes (since gdf_columns are not created until end)
    vector<string>		col_names;		// host: array of column names
} raw_csv_t;

//-- define the fields
typedef struct fields_info_ {
	int *				rec_id;			// on-device: the record index
	int *				col_id;			// on-device: the column index
	long *				start_idx;		// on-device: the starting bytes of the field
	long *				end_idx;		// on-device: the ending byte of the field - this could be the delimiter or eol
}  fields_info_t;



//
//---------------create and process ---------------------------------------------
//
gdf_error parseColArguments(const std::vector<std::string>  &col, const std::unordered_map<std::string, std::string> &dtypes, const char delim, raw_csv_t *d);
gdf_error updateRawCsv( const char * data, long num_bytes, raw_csv_t * csvData );
gdf_error allocateFieldInfoSpace(fields_info_t *, long num_rows, int num_col);
gdf_error allocateGdfDataSpace(gdf_column *);
gdf_dtype convertStringToDtype(std::string &dtype);

char *convertStringToChar(std::string str);

gdf_error freeCSVSpace(raw_csv_t * raw_csv);
gdf_error freeInfoSpace(fields_info_t * info);
gdf_error freeCsvData(char *data);

inline gdf_error checkError(gdf_error error, const char * txt) {
	if ( error != GDF_SUCCESS)
		std::cout << "ERROR:  " << error <<  "  in "  << txt;
	return error;
}

template<typename T>
gdf_error allocateTypeN(void *gpu, long N);

//
//---------------CUDA Kernel ---------------------------------------------
//
gdf_error launch_determineRecAndFields(raw_csv_t * data);  // breaks out fields and computed blocks - make main code cleaner

__global__ void determineRecAndFields(char *data, uint64_t * r_bits, uint64_t *f_bits, const char delim, long num_bytes, long num_bits, int * rec_count);

gdf_error launch_FieldBuilder(raw_csv_t * csv, fields_info_t *info);

__global__ void buildFieldList(
		char *		csv_data,
		long		chunks,
		long 		bytes,
		uint64_t * r_bits,
		long * 		offsets,
		int *		recPerChunck,
		char 		delim,
		int		num_row,
		int 		num_col,
		int *		rec_id,
		int *		col_id,
		long *		start_idx,
		long *		end_idx
		);


gdf_error launch_dataConvertColumn(raw_csv_t * raw_csv, fields_info_t *info, gdf_column * gdf, int col);

__global__ void convertCsvToGdf(
		char *		csv_data,
		char		delim,
		long 		entries,
		int 		col_id,
		gdf_dtype 	dtype,
		void *		gdf_data,
		gdf_valid_type *valid,
		int *		field_rec_id,
		int *		field_col_idx,
		long *		field_start_idx,
		long *		field_end_idx
		);


__device__ int findSetBit(int tid, uint64_t *f_bits, int x);

//
//---------------CUDA Valid (8 blocks of 8-bits) Bitmap Kernels ---------------------------------------------
//
__device__ int whichBitmap(int record) { return (record/8);  }
__device__ int whichBit(int bit) { return (bit % 8);  }
__device__ int checkBit(gdf_valid_type data, int bit) {

	gdf_valid_type bitMask[8] 		= {1, 2, 4, 8, 16, 32, 64, 128};

	return (data & bitMask[bit]);
}

__device__ int setBit(gdf_valid_type data, int bit) {

	gdf_valid_type bitMask[8] 		= {1, 2, 4, 8, 16, 32, 64, 128};

	return (data | bitMask[bit]);
}


//
//---------------Debug stuff (can be deleted) ---------------------------------------------
//
void printCheck(raw_csv_t * csvData, int start_idx, int num_blocks, const char * text);
void printGdfCheck(gdf_column * gdf, int start_data_idx, int num_records, const char * text);
void printInfoCheck(fields_info_t *info, int num_records, const char * text);
void printResults(gdf_column ** data, int num_col, int start_data_idx, int num_records);

//
//---------------old API  ---------------------------------------------
//
gdf_column ** readCSVbeta(
		const char * fileName, 										// in: the file to be loaded
		const char delimiter,										// in: the delimiter
		const std::vector<std::string> &col,						// in: the column names - ** NOTE ** the order of the names needs to map to the order of columns in the file
		const std::unordered_map<std::string, std::string> &dtypes	// in: the data types of each column.  THis does not need to be ordered
		);

//
//------------------------------------------------------------
//

/**
 * New API  - hack that just wrapper arguments and calls old API - will be fixed soon
 *
 */


gdf_column ** read_csv(
		const char 	*fileName, 			// in: the file to be loaded
		const char 	delimiter,			// in: the delimiter
		const int	num_cols,			// in: number of columns
		const char 	**col_names,			// in: ordered list of column names
		const char	**dtypes			// in: ordered list of dtypes
		)
{

	std::vector<std::string> col;
	std::unordered_map<std::string, std::string> dtype_map;


	for (int x = 0; x < num_cols; x++ ) {

		std::string k( col_names[x]);
		std::string v( dtypes[x] );

		col.push_back(k);
		dtype_map.insert( std::make_pair(k, v));
	}

	gdf_column **data = readCSVbeta(fileName, delimiter, col, dtype_map);

	return data;
}



/**
 * Release 0.0.0 Alpha code
 */
/**
 *
 */
gdf_column ** readCSVbeta(
		const char * fileName, 										// in: the file to be loaded
		const char delimiter,										// in: the delimiter
		const std::vector<std::string> &col,						// in: the column names - ** NOTE ** the order of the names needs to map to the order of columns in the file
		const std::unordered_map<std::string, std::string> &dtypes	// in: the data types of each column.  THis does not need to be ordered
		)
{

	gdf_error error = gdf_error::GDF_SUCCESS;

	//-----------------------------------------------------------------------------
	// create the CSV data structure - this will be filled in as the CSV data is processed.  Done first to validate data types
	raw_csv_t * raw_csv = new raw_csv_t;
	error = parseColArguments(col, dtypes, delimiter, raw_csv);
	checkError(error, "call to creatCsvDataStruct");

	//-----------------------------------------------------------------------------
	//---  load file -- using memory mapping
	struct stat     st;
	int				fd;
	void * 			mapData;
	size_t 			bytes;

	fd = open(fileName, O_RDONLY );
	if (fd < 0) {
		std::cerr << "Error opening file" << std::endl;
		//return GDF_C_ERROR;
		return NULL;
	}

	if (fstat(fd, &st)) {
		std::cerr <<  " cannot stat file: " << fileName << std::endl;
		//return GDF_C_ERROR;
		return NULL;
	}

	bytes = st.st_size;

	mapData = mmap(0, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapData == MAP_FAILED) {
    	close(fd);
		std::cerr << "Error mapping file" << std::endl;
		return NULL;
		//return GDF_C_ERROR;
    }

	//-----------------------------------------------------------------------------
	//---  create a structure to hold variables used to parse the CSV data
	error = updateRawCsv( (const char *)mapData, (long)bytes, raw_csv );
	//checkError(error, "call to createRawCsv");

	//-----------------------------------------------------------------------------
	// find the record and fields points (in bitmaps)
	error = launch_determineRecAndFields(raw_csv);
	//checkError(error, "call to Determine record offsets");

	//-----------------------------------------------------------------------------
	//--- Get the number of records
	thrust::device_ptr<int> tRecC(raw_csv->recPerChunck);
	raw_csv->num_records = thrust::reduce(tRecC, (tRecC + raw_csv->num_bits), 0, thrust::plus<int>());

	//-----------------------------------------------------------------------------
	//--- generate indexing (perform an inclusive scan)
	thrust::device_ptr<long> off(raw_csv->offsets);
    thrust::inclusive_scan(thrust::device, tRecC, (tRecC + raw_csv->num_bits), off);

	//printCheck(raw_csv, 0, 10, "after offset computation");

	//-----------------------------------------------------------------------------
	// create space for the list of field
	fields_info_t *info = new fields_info_t;
	error = allocateFieldInfoSpace(info, raw_csv->num_records, raw_csv->num_cols);
	//checkError(error, "call to allocateFieldInfoSpace");

	//-----------------------------------------------------------------------------
	// now create a list of all the field start and end points
	error = launch_FieldBuilder(raw_csv, info);
	//checkError(error, "call to launch_FieldBuilder");

	//printInfoCheck(info, 50, "field Info check");

	//-----------------------------------------------------------------------------
	// free up space that is no longer needed
	error = freeCSVSpace(raw_csv);
	//checkError(error, "freeing raw_csv_t space");

	//-----------------------------------------------------------------------------
	// last step is to copy over the data - this also fills in gdf_data with the columns

	//--- allocate space for the results
	gdf_column **cols = (gdf_column **)malloc( sizeof(gdf_column *) * raw_csv->num_cols);


	for (int col = 0; col < raw_csv->num_cols; col++) {

		gdf_column *gdf = (gdf_column *)malloc(sizeof(gdf_column) * 1);

		gdf->size		= raw_csv->num_records;
		gdf->dtype		= raw_csv->dtypes[col];
		gdf->null_count	= 0;						// will be filled in later

		//--- column name
		std::string str = raw_csv->col_names[col];
		int len = str.length() + 1;
		gdf->col_name = (char *)malloc(sizeof(char) * len);
		memcpy(gdf->col_name, str.c_str(), len);
		gdf->col_name[len -1] = '\0';

		// allocate the CUDA memory
		allocateGdfDataSpace(gdf);

		launch_dataConvertColumn(raw_csv, info, gdf, col);
		checkError(error, "call to launch_dataConvert " + col);

		cols[col] = gdf;
	}


	//-----------------------------------------------------------------------------
	//--- clean-up everything but the gdf_columns
	error = freeInfoSpace(info);
	checkError(error, "call to freeInfoSpace");
	delete info;

	error = freeCsvData(raw_csv->data);
	checkError(error, "call to cudaFree(raw_csv->data)" );
	delete raw_csv;

	//printResults(cols, raw_csv->num_cols, 0, 10);

	return cols;
}

//------------------------------------------------------------------------------------------------------------------------------

/**
 * Since the column names remain on the host (CPU) side, they are saved as a std::string rather than char *
 *
 * This creates the csv_data_t structure which in turn creates the basic gdf_coulmn structure
 *
 */
gdf_error parseColArguments(const std::vector<std::string>  &col_names, const std::unordered_map<std::string, std::string> &dtypes, const char delim, raw_csv_t *d)
{

	int num_col = col_names.size();						// how many columns are present

	d->num_cols		= num_col;
	d->num_records	= 0;
	d->delimiter	= delim;

	// start creating space for each column
	for ( int x = 0; x < num_col; x++) {

		std::string cn 	= col_names[x];
		std::string dtA = dtypes.find(cn)->second;
		gdf_dtype dt 	= convertStringToDtype( dtA );

		//std::cout << cn << " == "  << dtA << "  ==  " << dt << std::endl;

		if (dt == GDF_invalid || dt == GDF_STRING)
			return GDF_UNSUPPORTED_DTYPE;

		d->dtypes.push_back(dt);
		d->col_names.push_back(cn);
	}

	return gdf_error::GDF_SUCCESS;
}


/*
 * What is passed in is the data type as a string, need to convert that into gdf_dtype enum
 */
gdf_dtype convertStringToDtype(std::string &dtype) {

	gdf_dtype dt = GDF_invalid;

	if (dtype.compare( "str") == 0) {
			dt = GDF_STRING;
	}
	else if (dtype.compare( "date") == 0) {
			dt = GDF_DATE64;
	}
	else 	if (dtype.compare( "category") == 0) {
		dt = GDF_CATEGORY;
	}
	else if (dtype.compare( "float") == 0)  {
		dt = GDF_FLOAT32;
	}
	else if (dtype.compare( "float32") == 0) {
		dt = GDF_FLOAT32;
	}
	else if (dtype.compare( "float64") == 0)  {
		dt = GDF_FLOAT64;
	}
	else if (dtype.compare( "double") == 0)  {
		dt = GDF_FLOAT64;
	}
	else if (dtype.compare( "short") == 0)  {
		dt = GDF_INT16;
	}
	else if (dtype.compare( "int") == 0)  {
		dt = GDF_FLOAT32;
	}
	else if (dtype.compare( "int64") == 0)  {
		dt = GDF_INT64;
	}
	else if (dtype.compare( "long") == 0)  {
		dt = GDF_INT64;
	}
	else {
			dt = GDF_invalid;
	}

	return dt;
}


/*
 * Create the raw_csv_t structure and allocate space on the GPU
 */
gdf_error updateRawCsv( const char * data, long num_bytes, raw_csv_t * raw ) {

	int num_bits = (num_bytes + 63) / 64;

	CUDA_TRY(cudaMallocManaged ((void**)&raw->data, 		(sizeof(char)		* num_bytes)));
	CUDA_TRY(cudaMallocManaged ((void**)&raw->rec_bits, 	(sizeof(uint64_t)	* num_bits)));
	CUDA_TRY(cudaMallocManaged ((void**)&raw->field_bits, 	(sizeof(uint64_t)	* num_bits)));
	CUDA_TRY(cudaMallocManaged ((void**)&raw->recPerChunck,	(sizeof(int) 		* num_bits)) );
	CUDA_TRY(cudaMallocManaged ((void**)&raw->offsets, 		((sizeof(long)		* num_bits) + 2)) );

	CUDA_TRY(cudaMemcpy(raw->data, data, num_bytes, cudaMemcpyHostToDevice));

	CUDA_TRY( cudaMemset(raw->rec_bits, 	0, (sizeof(uint64_t) 	* num_bits)) );
	CUDA_TRY( cudaMemset(raw->field_bits, 	0, (sizeof(uint64_t) 	* num_bits)) );
	CUDA_TRY( cudaMemset(raw->recPerChunck, 0, (sizeof(int) 		* num_bits)) );
	CUDA_TRY( cudaMemset(raw->offsets, 		0, ((sizeof(long) 		* num_bits) + 2)) );

	raw->num_bytes = num_bytes;
	raw->num_bits  = num_bits;

	return GDF_SUCCESS;
}


gdf_error allocateFieldInfoSpace(fields_info_t *info, long num_rows, int num_col)
{
	uint64_t N = (uint64_t)num_rows * (uint64_t)num_col;

	CUDA_TRY(cudaMallocManaged ((void**)&info->rec_id, 		(sizeof(int) * N)));
	CUDA_TRY(cudaMallocManaged ((void**)&info->col_id, 		(sizeof(int) * N)));
	CUDA_TRY(cudaMallocManaged ((void**)&info->start_idx, 	(sizeof(long) * N)));
	CUDA_TRY(cudaMallocManaged ((void**)&info->end_idx, 	(sizeof(long) * N)));

	return GDF_SUCCESS;
}


/*
 * For each of the gdf_cvolumns, create the on-device space.  the on-host fields should already be filled in
 */
gdf_error allocateGdfDataSpace(gdf_column *gdf) {

	long N = gdf->size;
	int num_bitmaps = (N + 7) / 8;			// 8 bytes per bitmap

	//--- allocate space for the valid bitmaps
	CUDA_TRY(cudaMallocManaged(&gdf->valid, (sizeof(gdf_valid_type) 	* num_bitmaps)));
	CUDA_TRY(cudaMemset(gdf->valid, 0, (sizeof(gdf_valid_type) 	* num_bitmaps)) );

	//--- Allocate space for the data
	switch(gdf->dtype) {
		case gdf_dtype::GDF_INT8:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int8_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int8_t) 	* N)) );
			break;
		case gdf_dtype::GDF_INT16:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int16_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int16_t) 	* N)) );
			break;
		case gdf_dtype::GDF_INT32:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int32_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int32_t) 	* N)) );
			break;
		case gdf_dtype::GDF_INT64:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int64_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int64_t) 	* N)) );
			break;
		case gdf_dtype::GDF_FLOAT32:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(float) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(float) 	* N)) );
			break;
		case gdf_dtype::GDF_FLOAT64:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(double) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(double) 	* N)) );
			break;
		case gdf_dtype::GDF_DATE64:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(gdf_date64) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(gdf_date64) 	* N)) );
			break;
		case gdf_dtype::GDF_CATEGORY:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(gdf_categoty) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(gdf_categoty) 	* N)) );
			break;
		case gdf_dtype::GDF_STRING:
			return GDF_UNSUPPORTED_DTYPE;
		default:
			return GDF_UNSUPPORTED_DTYPE;
	}

	return gdf_error::GDF_SUCCESS;
}


gdf_error freeCSVSpace(raw_csv_t * raw_csv)
{
	CUDA_TRY(cudaFree(raw_csv->rec_bits));
	CUDA_TRY(cudaFree(raw_csv->field_bits));
	CUDA_TRY(cudaFree(raw_csv->recPerChunck));
	CUDA_TRY(cudaFree(raw_csv->offsets));

	return gdf_error::GDF_SUCCESS;
}

gdf_error freeInfoSpace(fields_info_t * info)
{
	CUDA_TRY(cudaFree(info->rec_id));
	CUDA_TRY(cudaFree(info->col_id));
	CUDA_TRY(cudaFree(info->start_idx));
	CUDA_TRY(cudaFree(info->end_idx));

	return gdf_error::GDF_SUCCESS;

}

gdf_error freeCsvData(char *data)
{
	CUDA_TRY(cudaFree(data));

	return gdf_error::GDF_SUCCESS;

}


//----------------------------------------------------------------------------------------------------------------
//				CUDA Kernels
//----------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------
gdf_error launch_determineRecAndFields(raw_csv_t * csvData) {

	char 		*data 		= csvData->data;
	uint64_t 	*r_bits		= csvData->rec_bits;
	uint64_t 	*f_bits		= csvData->field_bits;
	long 		num_bytes	= csvData->num_bytes;
	int			*rec_count	= csvData->recPerChunck;
	long 		numBitmaps 	= csvData->num_bits;
	char		delim		= csvData->delimiter;


	/*
	 * Each bi map is for a 64-byte chunk, and technically we could do a thread per 64-bytes.
	 * However, that doesn't seem efficient.

	 *      Note: could do one thread per byte, but that would require a lock on the bit map
	 *
	 */
	int threads 	= 1024;

	// Using the number of bitmaps as the size - data index is bitmap ID * 64
	int blocks = (numBitmaps + (threads -1)) / threads ;

	determineRecAndFields <<< blocks, threads >>> (data, r_bits, f_bits, delim, num_bytes, numBitmaps, rec_count);

	CUDA_TRY(cudaGetLastError());
	return GDF_SUCCESS;
}


__global__ void determineRecAndFields(char *data, uint64_t * r_bits, uint64_t *f_bits, const char delim, long num_bytes, long num_bits, int * rec_count) {

	// thread IDs range per block, so also need the block id
	int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if ( tid >= num_bits)
		return;

	// data ID - multiple of 64
	long did = tid * 64;

	char *raw = (data + did);

	int byteToProcess = ((did + 64) < num_bytes) ? 64 : (num_bytes - did);

	// process the data
	int x = 0;
	for (x = 0; x < byteToProcess; x++) {

		// fields
		if (raw[x] == delim) {
			f_bits[tid] |= 1UL << x;

		} else {
			// records
			if (raw[x] == '\n') {
				r_bits[tid] |= 1UL << x;

			}	else if (raw[x] == '\r' && raw[x +1] == '\n') {
				x++;
				r_bits[tid] |= 1UL << x;
			}
		}
	}

	// save the number of records detected within this block
	uint64_t bitmap = r_bits[tid];
	while (bitmap)
	{
		rec_count[tid] += bitmap & 1;
		bitmap >>= 1;
	 }
}


//----------------------------------------------------------------------------------------------------------------


gdf_error launch_FieldBuilder(raw_csv_t * csv, fields_info_t *info)
{
	int threads 	= 1024;

	// Using the number of bitmaps as the size - data index is bitmap ID * 64
	int blocks = (csv->num_bits + (threads -1)) / threads ;

	buildFieldList <<< blocks, threads >>>(
			csv->data,
			csv->num_bits,
			csv->num_bytes,
			csv->rec_bits,
			csv->offsets,
			csv->recPerChunck,
			csv->delimiter,
			csv->num_records,
			csv->num_cols,
			info->rec_id,
			info->col_id,
			info->start_idx,
			info->end_idx
			);

	CUDA_TRY(cudaGetLastError());
	return GDF_SUCCESS;
}

__global__ void buildFieldList(
		char *		csv_data,
		long		num_chunks,
		long 		bytes,
		uint64_t * r_bits,
		long * 		offsets,
		int *		recPerChunck,
		const char delim,
		int 		num_row,
		int 		num_col,
		int *		field_rec_id,
		int *		field_col_idx,
		long *		field_start_idx,
		long *		field_end_idx
		)
{
	int bit_idx				= 1;					// used to find offset


	// thread IDs range per block, so also need the block id
	int	tid  = threadIdx.x + (blockDim.x * blockIdx.x);		// this is the data chunk index;  NOTE not the record ID

	// we can have more threads than data, make sure we are not past the end of the data
	if ( tid >= num_chunks)
		return;

	int records_to_process 	= recPerChunck[tid];	// how many records are within this 64-byte chunk

	// the first record is special since it does not start with a record bitmap, it also does not show up in recPerChunck count
	if ( tid == 0) {
		++records_to_process;	// Increment the number of records to process
		bit_idx = 0;			// look for the 0 occurrence - just returns
	}

	if (records_to_process == 0)						// There are no records within this chunk - nothing to process
		return;

	//
	//-  define some variables needed
	//
	long did 				= tid * 64;				// index into the raw csv data - process in  multiple of 64
	int rec_id 				= offsets[tid];			// The (first) record index
	int start_offset 		= 0;					// starting point within the data - will be updated during processing
	int end_offset			= 0;					// end point - updated during processing
	uint64_t field_s		= 0;
	uint64_t field_e 		= 0;
	uint64_t rec_start_idx 	= 0;
	uint64_t rec_end_idx 	= 0;
	int 	col				= 0;

	start_offset = findSetBit(tid, r_bits, bit_idx);

	if ((did + start_offset + 1) >= bytes)
		return;		// this is just the end of file indicator

	for ( int n = 0; n < records_to_process; n++) {
		// never go out of bounds
		if (rec_id >= num_row) break;

		++bit_idx;
		end_offset 		= findSetBit(tid, r_bits, bit_idx);

		rec_start_idx 	= did + start_offset + 1;				// the start_offset is at the record bit point, move over 1
		rec_end_idx 	= did + end_offset;						// the end is at the bit point
		field_s 		= rec_start_idx;
		field_e 		= rec_start_idx;
		col				= 0;				// this will be incremented as data is processed


		while (field_e < (rec_end_idx +1) ) {

			/*
			 * found a field delimiter, process the data within this block
			 */
			if (csv_data[field_e] == delim || field_e == rec_end_idx)		// found a block of data to process
			{
				//if (field_e == rec_end_idx)
				//	field_e++;

				int field_bytes = field_e - field_s;

				uint64_t idx =  (rec_id *num_col) + col;
				field_rec_id[idx] 	= rec_id;
				field_col_idx[idx] 	= col;

				if ( field_bytes > 0) {
					field_start_idx[idx]= field_s;
					field_end_idx[idx]	= field_e;
				} else {
					field_start_idx[idx]= -1;		// indicates an invalid field
					field_end_idx[idx]	= -1;
				}

				field_s = field_e +1;
				col++;
			}	// end of if delimiter check

			++field_e;		// move the end over one space
		}

		// process data between start(+1) and end (-1)

		start_offset = end_offset +1;
		++rec_id;
	}

}




//----------------------------------------------------------------------------------------------------------------


gdf_error launch_dataConvertColumn(raw_csv_t * raw_csv, fields_info_t *info, gdf_column * gdf, int col) {

	int num_cols 	 = raw_csv->num_cols;
	long num_entries = raw_csv->num_records * num_cols;

	int threads 	= 1024;
	int blocks 		= ( num_entries + (threads -1)) / threads ;

	convertCsvToGdf <<< blocks, threads >>>(
		raw_csv->data,
		raw_csv->delimiter,
		num_entries,
		col,
		gdf->dtype,
		gdf->data,
		gdf->valid,
		info->rec_id,
		info->col_id,
		info->start_idx,
		info->end_idx
	);


	return GDF_SUCCESS;
}





/*
 * Data is processed in 64-bytes chunks - so the number of total threads (tid) is equal to the number of bitmaps
 * a thread will start processing at the start of a record and end of the end of a record, even if it crosses
 * a 64-byte boundary
 *
 * tid = 64-byte index
 * did = offset into character data
 * offset = record index (starting if more that 1)
 *
 */
__global__ void convertCsvToGdf(
		char *		raw_csv,
		char		delim,
		long 		num_entries,
		int 		col_id_to_process,
		gdf_dtype 	dtype,
		void *		gdf_data,
		gdf_valid_type *valid,
		int *		field_rec_id,
		int *		field_col_idx,
		long *		field_start_idx,
		long *		field_end_idx
		)
{

	// thread IDs range per block, so also need the block id
	long	tid  = threadIdx.x + (blockDim.x * blockIdx.x);		// this is entry into the field array - tid is an elements within the num_entries array

	// we can have more threads than data, make sure we are not past the end of the data
	if ( tid >= num_entries)
		return;

	int rec_id			= field_rec_id[tid];
	int col_id			= field_col_idx[tid];
	long start_idx		= field_start_idx[tid];
	long end_idx		= field_end_idx[tid] -1;

	if ( col_id != col_id_to_process)
		return;

	if ( start_idx != -1) {
		if (raw_csv[end_idx] == delim)
			--end_idx;

		switch(dtype) {
			case gdf_dtype::GDF_INT8:
			{
				int8_t *gdf_out = (int8_t *)gdf_data;
				gdf_out[rec_id] = convertStrtoInt<int8_t>(raw_csv, start_idx, end_idx);
			}
				break;
			case gdf_dtype::GDF_INT16: {
				int16_t *gdf_out = (int16_t *)gdf_data;
				gdf_out[rec_id] = convertStrtoInt<int16_t>(raw_csv, start_idx, end_idx);
			}
				break;
			case gdf_dtype::GDF_INT32:
			{
				int32_t *gdf_out = (int32_t *)gdf_data;
				gdf_out[rec_id] = convertStrtoInt<int32_t>(raw_csv, start_idx, end_idx);
			}
				break;
			case gdf_dtype::GDF_INT64:
			{
				int64_t *gdf_out = (int64_t *)gdf_data;
				gdf_out[rec_id] = convertStrtoInt<int64_t>(raw_csv, start_idx, end_idx);
			}
				break;
			case gdf_dtype::GDF_FLOAT32:
			{
				float *gdf_out = (float *)gdf_data;
				gdf_out[rec_id] = convertStrtoFloat<float>(raw_csv, start_idx, end_idx);
			}
				break;
			case gdf_dtype::GDF_FLOAT64:
			{
				double *gdf_out = (double *)gdf_data;
				gdf_out[rec_id] = convertStrtoFloat<double>(raw_csv, start_idx, end_idx);
			}
				break;
			case gdf_dtype::GDF_DATE64:
			{
				gdf_date64 *gdf_out = (gdf_date64 *)gdf_data;
				gdf_out[rec_id] = convertStrtoDate(raw_csv, start_idx, end_idx);
			}

				break;
			case gdf_dtype::GDF_CATEGORY:
			{
				gdf_categoty *gdf_out = (gdf_categoty *)gdf_data;
				gdf_out[rec_id] = convertStrtoHash(raw_csv, start_idx, end_idx, HASH_SEED);
			}
				break;
			case gdf_dtype::GDF_STRING:
				break;
			default:
				break;
		}

		// set the valid bitmap - all bits were set to 0 to start
		int bitmapIdx 	= whichBitmap(rec_id);  		// which bitmap
		int bitIdx		= whichBit(rec_id);				// which bit - over an 8-bit index
		gdf_valid_type	bitmap	= valid[bitmapIdx];		// get the bitmap
		valid[bitmapIdx] = setBit(bitmap, bitIdx);		// set the bit
	}

}



/*
 * Return which bit is set
 * x is the occurrence: 1 = first, 2 = seconds, ...
 */
__device__ int findSetBit(int tid, uint64_t *r_bits, int x) {

	int idx = tid;

	if ( x == 0 )
		return -1;

	int withinBitCount = 0;
	int offset = 0;
	int found  = 0;

	uint64_t bitmap = r_bits[idx];

	while (found != x)
	{
		if(bitmap == 0)
		{
			idx++;
			bitmap = r_bits[idx];
			offset += 64;
			withinBitCount = 0;
		}

		if ( bitmap & 1 ) {
			found++;			//found a set bit
		}

		bitmap >>= 1;
		++withinBitCount;
	 }

	offset += withinBitCount -1;


	return offset;
}





//---------------------------------------------------------------------------------------------------------------
//
//			Debug functions below this point
//
//---------------------------------------------------------------------------------------------------------------

void printCheck(raw_csv_t * csvData, int start_data_idx, int num_blocks, const char * text) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Checking (dependent on Unified Memory) - " 		<< text << std::endl;
	std::cout << "\tNumber of Bytes   - " << csvData->num_bytes 	<< std::endl;
	std::cout << "\tNumber of Bitmaps - " << csvData->num_bits  	<< std::endl;
	std::cout << "\tNumber of Records - " << csvData->num_records  	<< std::endl;


	char * data = csvData->data;

	int data_idx 	= start_data_idx;

	if ( data_idx != 0)
		while ( data_idx % 64 != 0 )
			data_idx++;

	std::cout << "\tStarting Index specified - " << start_data_idx << "  adjusted to " << data_idx << std::endl;

	int bitmap_idx  = data_idx / 64;

	std::cout << "\tStarting Bitmap Index - " << bitmap_idx  << std::endl;
	std::cout << "[data, bit] =>                 64 bytes of data                 \t rec_bit   field_bits    Record counts"  << std::endl;


	for ( int loop = 0; loop < num_blocks; loop++) {
		std::cout << "[" << std::setw(6) << data_idx << " ,  " << std::setw(6) << bitmap_idx << "] =>  ";

		 for ( int x = 0; x < 64; x++) {

			if ( data_idx < csvData->num_bytes) {
				if (data[data_idx] == '\n')
					std::cout << "\033[1m\033[31mNL\033[0m ";
				else
					std::cout << data[data_idx] << " ";
			}

			++data_idx;
		 }

		std::cout << " =>  " << std::setw(25) << csvData->rec_bits[bitmap_idx];
		std::cout << "\t" << std::setw(25) << csvData->field_bits[bitmap_idx];
		std::cout << "\t" << std::setw(2) << csvData->recPerChunck[bitmap_idx];
		std::cout << "\t" << std::setw(2) << csvData->offsets[bitmap_idx];
		std::cout << std::endl;

		++bitmap_idx;
	}

	std::cout << "--------------------------------\n\n";

}


void printGdfCheck(gdf_column * gdf, int start_data_idx, int num_records, const char * text) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Checking (dependent on Unified Memory) - " << text << std::endl;
	std::cout << "\tCol: " << gdf->col_name  <<  " and data type of " << gdf->dtype << std::endl;

	for ( int x = 0; x < num_records; x++) {
		switch(gdf->dtype) {
			case gdf_dtype::GDF_INT8:
			{
				int8_t *gdf_out = (int8_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_INT16: {
				int16_t *gdf_out = (int16_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_INT32:
			{
				int32_t *gdf_out = (int32_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_INT64:
			{
				int64_t *gdf_out = (int64_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_FLOAT32:
			{
				float *gdf_out = (float *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_FLOAT64:
			{
				double *gdf_out = (double *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_DATE64:
				break;
			case gdf_dtype::GDF_CATEGORY:
				break;
			case gdf_dtype::GDF_STRING:
				break;
			default:
				break;
		}
	}

	std::cout << "--------------------------------\n\n";

}


void printResults(gdf_column ** data, int num_col, int start_data_idx, int num_records) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Printing Results (dependent on Unified Memory) " << std::endl;

	for ( int c = 0;  c < num_col; c++)
	{
		gdf_column *gdf = data[c];

		std::cout << std::setw(2) << c << " =>  " << std::setw(50) << gdf->col_name << "\t" << gdf->dtype << "\t" << gdf->size << "\t" << gdf->null_count << std::endl;
	}
	std::cout << std::endl;

	long x = 0;
	for ( int i = 0; i < num_records; i++) {
		x = start_data_idx + i;

		std::cout << "\tRec[ " << x << "] is:  ";


		for ( int c = 0;  c < num_col; c++)
		{
			gdf_column *gdf = data[c];


			switch(gdf->dtype) {
				case gdf_dtype::GDF_INT8:
				{
					int8_t *gdf_out = (int8_t *)gdf->data;
					std::cout << "\t"  << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_INT16: {
					int16_t *gdf_out = (int16_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_INT32:
				{
					int32_t *gdf_out = (int32_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_INT64:
				{
					int64_t *gdf_out = (int64_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_FLOAT32:
				{
					float *gdf_out = (float *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_FLOAT64:
				{
					double *gdf_out = (double *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_DATE64:
				{
					int64_t *gdf_out = (int64_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
				break;
				case gdf_dtype::GDF_CATEGORY:
				{
					int32_t *gdf_out = (int32_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_STRING:
					break;
				default:
					break;
			}
		}
		std::cout << std::endl;
	}

	std::cout << "--------------------------------\n\n";

}



void printInfoCheck(fields_info_t *info, int num_records, const char * text) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Checking (dependent on Unified Memory) - " << text << std::endl;

	std::cout << "\tRec Id\tCol Id\tStart Idx\tEnd Idx" << std::endl;

	for ( int x = 0; x < num_records; x++) {
		std::cout << "\t" << info->rec_id[x] << "\t" << info->col_id[x] << "\t" << info->start_idx[x] << "\t" << info->end_idx[x]   << std::endl;
	}

}






















