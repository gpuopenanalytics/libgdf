#pragma once

typedef size_t gdf_size_type;
typedef gdf_size_type gdf_index_type;

/**
 * @brief A bit-holder type, used for indicating whether some column elements
 * are null or not. If the corresponding element is null, its bit will be 0;
 * otherwise the value is 1 (a "valid" element)
 */
typedef unsigned char gdf_valid_type;
typedef	long	gdf_date64;
typedef	int		gdf_date32;
typedef	int		gdf_category;

/**
 * @brief  Possible data types for a @ref `gdf_column`
 */
typedef enum {
    GDF_invalid=0,
    GDF_INT8,
    GDF_INT16,
    GDF_INT32,
    GDF_INT64,
    GDF_FLOAT32,
    GDF_FLOAT64,
    GDF_DATE32,   	///< int32_t days since the UNIX epoch
    GDF_DATE64,   	///< int64_t milliseconds since the UNIX epoch
    GDF_TIMESTAMP,	///<  Exact timestamp encoded with int64 since UNIX epoch (Default unit millisecond)
    GDF_CATEGORY,
    GDF_STRING,
    N_GDF_TYPES, 	/* additional types should go BEFORE N_GDF_TYPES */
} gdf_dtype;


/**
 * @brief  Possible return values from libgdf functions.
 *
 * @note All error codes in this enum have corresponding descriptions
 * available via the @ref `gdf_error_get_name` function
 */
typedef enum {
    GDF_SUCCESS=0,                
    GDF_CUDA_ERROR,                   ///< Error occurred in a CUDA call
    GDF_UNSUPPORTED_DTYPE,            ///< The data type of the gdf_column is unsupported
    GDF_COLUMN_SIZE_MISMATCH,         ///< Two columns that should be the same size aren't the same size
    GDF_COLUMN_SIZE_TOO_BIG,          ///< Size of column is larger than the max supported size
    GDF_DATASET_EMPTY,                ///< An input column is either null or has size 0, when it must have data
    GDF_VALIDITY_MISSING,             ///< gdf_column's validity bitmask is null
    GDF_VALIDITY_UNSUPPORTED,         ///< The requested GDF operation does not support validity bitmask handling, and one of the input columns has the valid bits enabled
    GDF_INVALID_API_CALL,             ///< The arguments passed into the function were invalid
    GDF_JOIN_DTYPE_MISMATCH,          ///< Data type mismatch between corresponding columns in  left/right tables in the Join function
    GDF_JOIN_TOO_MANY_COLUMNS,        ///< Too many columns were passed in for the requested join operation
    GDF_DTYPE_MISMATCH,               ///< Type mismatch between columns that should be the same type
    GDF_UNSUPPORTED_METHOD,           ///< The method requested to perform an operation was invalid or unsupported (e.g., hash vs. sort)
    GDF_INVALID_AGGREGATOR,           ///< Invalid aggregator was specified for a group-by operation
    GDF_INVALID_HASH_FUNCTION,        ///< Invalid hash function was selected
    GDF_PARTITION_DTYPE_MISMATCH,     ///< Data type mismatch between columns of input/output in the hash partition function
    GDF_HASH_TABLE_INSERT_FAILURE,    ///< Failed to insert to hash table, likely because its full
    GDF_UNSUPPORTED_JOIN_TYPE,        ///< The type of join requested is unsupported
    GDF_UNDEFINED_NVTX_COLOR,         ///< The requested color used to define an NVTX range is not defined
    GDF_NULL_NVTX_NAME,               ///< The requested name for an NVTX range cannot be a null ptr
    GDF_C_ERROR,				      ///< C error not related to CUDA
    GDF_FILE_ERROR,   				  ///< Error processing the specified file
} gdf_error;

/**
 * @brief Possible hash functions for use in Joins, partitioning, and other operations libgdf provides.
 */
typedef enum {
    GDF_HASH_MURMUR3=0, ///< Murmur3 hash function; see @url https://en.wikipedia.org/wiki/MurmurHash#MurmurHash3
    GDF_HASH_IDENTITY,  ///< Identity hash function that simply returns the key to be hashed
} gdf_hash_func;

/**
 * @brief The resolution, or unit, for durations of time, used alongside duration values
 */
typedef enum {
	TIME_UNIT_NONE=0, ///< time unit is undefined/unknown; this is the implicit default
	TIME_UNIT_s,      ///< seconds
	TIME_UNIT_ms,     ///< milliseconds (10^{-3} seconds)
	TIME_UNIT_us,     ///< microseconds (10^{-6} seconds)
	TIME_UNIT_ns      ///< nanoseconds  (10^{-9} seconds)
} gdf_time_unit;

/**
 * @brief Potential Auxiliary information regarding a datum in a libgdf column.
 *
 * @note held either at the single-element or whole-column level.
 *
 */
typedef struct {
	gdf_time_unit time_unit;
	// here we can also hold info for decimal datatype or any other datatype that requires additional information
} gdf_dtype_extra_info;

/**
 * @brief The fundamental, columnar format of data the GDF library works with.
 *
 * A gdf_column_ may originate in a RDBMS-like schema table; it may be the intermediary result
 * within an execution plan; or it may be the result of non-DBMS related computation.
 */
 typedef struct gdf_column_{
     void *data;
         ///< Type-erased pointer to the column's raw data - which is a consecutive sequence
         ///< with no gaps of elements of the type represented by `dtype`.
         ///<
         ///< @todo There are currently no formal alignment requirements, but it seems the
         ///< implementation may implicitly be assuming alignment to the size of the relevant
         ///< type.
         ///< @todo Can this be NULL? What about after "construction"?
         ///< @todo Is this always in device memory?

     gdf_valid_type *valid;
         ///< a pseudo-column of `size` bits, packed into bytes (with in-byte order from
         ///< the least significant to the most significant bit), indicating whether
         ///< the column element is null (bit value is 0) or not null (bit value is 1;
         ///< a "valid" element)
         ///<
         ///< @todo There are currently no formal alignment requirements, but it seems the
         ///< implementation may implicitly be assuming alignment to the size of the relevant
         ///<
         ///< @todo Is this expressly forbidden from being NULL in the case of a 0 null-count?

     gdf_size_type size;
         ///< The number of column elements (_not_ their total size in bytes, _nor_ the
         ///< size of an individual element)
         ///<
         ///< @todo is it allocated capacity or size in use?

     gdf_dtype dtype;
         ///< An indicator of the column's data type, for type un-erasure

     gdf_size_type null_count;
         ///< The number of null elements in the column, which is
         ///< also the number of 0 bits in the `valid` pseudo-column
         ///< (within the range of valid bits, i.e. 0..size-1 )

     gdf_dtype_extra_info dtype_info;
         ///< Additional information qualifying the data type

     char *	col_name;
     	 ///< The column's name - a NUL-terminated string in host memory

 } gdf_column;

/** 
 * @brief  These enums indicate which method is to be used for an operation.
 * For example, it is used to select between the hash-based vs. sort-based implementations
 * of the Join operation.
 */
typedef enum {
  GDF_SORT = 0,   ///< Indicates that the sort-based implementation of the function will be used
  GDF_HASH,       ///< Indicates that the hash-based implementation of the function will be used

  // New enum values should be added above this line
  N_GDF_METHODS,
} gdf_method;

typedef enum {
  GDF_QUANT_LINEAR =0,
  GDF_QUANT_LOWER,
  GDF_QUANT_HIGHER,
  GDF_QUANT_MIDPOINT,
  GDF_QUANT_NEAREST,

  // New enum values should be added above this line
  N_GDF_QUANT_METHODS,
} gdf_quantile_method;


/** 
 * @brief Possible aggregation (=reduction) function which may be performed on a
 * column, or sequence of aggregation columns, by a GroupBy operation
 *
 * Also @ref window_reduction_type .
 */
typedef enum {
  GDF_SUM = 0,        ///< Computes the sum of all values in the aggregation column
  GDF_MIN,            ///< Computes minimum value in the aggregation column
  GDF_MAX,            ///< Computes maximum value in the aggregation column
  GDF_AVG,            ///< Computes arithmetic mean of all values in the aggregation column
  GDF_COUNT,          ///< Computes histogram of the occurance of each key in the GroupBy Columns
  GDF_COUNT_DISTINCT, ///< Counts the number of distinct keys in the GroupBy columns

  // New enum values should be added above this line
  N_GDF_AGG_OPS,      ///< The total number of aggregation operations.
} gdf_agg_op;


/** 
 * @brief  Colors for use with NVTX ranges.
 *
 * These enumerations are the available pre-defined colors for use with
 * user-defined NVTX ranges.
 */
typedef enum {
  GDF_GREEN = 0, 
  GDF_BLUE,
  GDF_YELLOW,
  GDF_PURPLE,
  GDF_CYAN,
  GDF_RED,
  GDF_WHITE,
  GDF_DARK_GREEN,
  GDF_ORANGE,

  // New enum values should be added above this line
  GDF_NUM_COLORS,
} gdf_color;

/** 
 * @brief  Information about how an operation should be performed and about its input
 */
typedef struct gdf_context_{
  int flag_sorted;        ///< Indicates if the input data is sorted. 0 = No, 1 = yes
  gdf_method flag_method; ///< The method to be used for the operation (e.g., sort vs hash)
  int flag_distinct;      ///< for COUNT: DISTINCT = 1, else = 0
  int flag_sort_result;   ///< When method is GDF_HASH, 0 = result is not sorted, 1 = result is sorted
  int flag_sort_inplace;  ///< 0 = No sort in place allowed, 1 = else
} gdf_context;

struct _OpaqueIpcParser;
typedef struct _OpaqueIpcParser gdf_ipc_parser_type;


struct _OpaqueRadixsortPlan;
typedef struct _OpaqueRadixsortPlan gdf_radixsort_plan_type;


struct _OpaqueSegmentedRadixsortPlan;
typedef struct _OpaqueSegmentedRadixsortPlan gdf_segmented_radixsort_plan_type;


typedef enum{
	GDF_ORDER_ASC,  ///< Ascending order
	GDF_ORDER_DESC  ///< Descending order
} order_by_type;

typedef enum{
	GDF_EQUALS,
	GDF_NOT_EQUALS,
	GDF_LESS_THAN,
	GDF_LESS_THAN_OR_EQUALS,
	GDF_GREATER_THAN,
	GDF_GREATER_THAN_OR_EQUALS
} gdf_comparison_operator;

typedef enum{
	GDF_WINDOW_RANGE,
	GDF_WINDOW_ROW
} window_function_type;

/**
 * @brief Possible aggregation (=reduction) function which may be performed on a
 * window in a column, or sof aggregation columns, by a GroupBy operation
 *
 * Also, see @ref gdf_agg_op .
 */
typedef enum{
	GDF_WINDOW_AVG,
	GDF_WINDOW_SUM,
	GDF_WINDOW_MAX,
	GDF_WINDOW_MIN,
	GDF_WINDOW_COUNT,
	GDF_WINDOW_STDDEV,
	GDF_WINDOW_VAR //variance
} window_reduction_type;
