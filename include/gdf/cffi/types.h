typedef size_t gdf_size_type;
typedef gdf_size_type gdf_index_type;
typedef unsigned char gdf_valid_type;

typedef enum {
    GDF_invalid=0,
    GDF_INT8,
    GDF_INT16,
    GDF_INT32,
    GDF_INT64,
    GDF_UINT8,
    GDF_UINT16,
    GDF_UINT32,
    GDF_UINT64,
    GDF_FLOAT32,
    GDF_FLOAT64,
    GDF_DATE32,   // int32_t days since the UNIX epoch
    GDF_DATE64,   // int64_t milliseconds since the UNIX epoch
    GDF_TIMESTAMP,// Exact timestamp encoded with int64 since UNIX epoch (Default unit millisecond)
    N_GDF_TYPES, /* additional types should go BEFORE N_GDF_TYPES */
} gdf_dtype;

union gdf_data {
    void*    invd;
    int8_t   si08;
    int16_t  si16;
    int32_t  si32;
    int64_t  si64;
    uint8_t  ui08;
    uint16_t ui16;
    uint32_t ui32;
    uint64_t ui64;
    float    fp32;
    double   fp64;
    int32_t  dt32;  // GDF_DATE32
    int64_t  dt64;  // GDF_DATE64
    int64_t  tmst;  // GDF_TIMESTAMP
};

typedef enum {
    GDF_SUCCESS=0,
    GDF_CUDA_ERROR,
    GDF_UNSUPPORTED_DTYPE,
    GDF_COLUMN_SIZE_MISMATCH,
    GDF_COLUMN_SIZE_TOO_BIG,
    GDF_DATASET_EMPTY,
    GDF_VALIDITY_MISSING,
    GDF_VALIDITY_UNSUPPORTED,
    GDF_JOIN_DTYPE_MISMATCH,
    GDF_JOIN_TOO_MANY_COLUMNS,
    GDF_UNSUPPORTED_METHOD,
} gdf_error;

typedef enum {
    GDF_HASH_MURMUR3=0,
} gdf_hash_func;

typedef enum {
	TIME_UNIT_NONE=0, // default (undefined)
	TIME_UNIT_s,   // second
	TIME_UNIT_ms,  // millisecond
	TIME_UNIT_us,  // microsecond
	TIME_UNIT_ns   // nanosecond
} gdf_time_unit;

typedef struct {
	gdf_time_unit time_unit;
	// here we can also hold info for decimal datatype or any other datatype that requires additional information
} gdf_dtype_extra_info;

struct gdf_scalar {
    gdf_data  data;
    gdf_dtype dtype;
};

typedef struct gdf_column_{
    void *data;
    gdf_valid_type *valid;
    gdf_size_type size;
    gdf_dtype dtype;
    gdf_dtype_extra_info dtype_info;
} gdf_column;

typedef enum {
  GDF_SORT = 0,
  GDF_HASH,
  N_GDF_METHODS,  /* additional methods should go BEFORE N_GDF_METHODS */
} gdf_method;

typedef enum {
  GDF_SUM = 0,
  GDF_MIN,
  GDF_MAX,
  GDF_AVG,
  GDF_COUNT,
  GDF_COUNT_DISTINCT,
  N_GDF_AGG_OPS, /* additional aggregation ops should go BEFORE N_GDF_... */
} gdf_agg_op;


enum gdf_binary_operator {
    GDF_ADD,
    GDF_SUB,
    GDF_MUL,
    GDF_DIV,
    GDF_TRUE_DIV,
    GDF_FLOOR_DIV,
    GDF_MOD,
    GDF_POW,
    //GDF_COMBINE,
    GDF_COMBINE_FIRST,
    GDF_ROUND,
    GDF_EQUAL,
    GDF_NOT_EQUAL,
    GDF_LESS,
    GDF_GREATER,
    GDF_LESS_EQUAL,
    GDF_GREATER_EQUAL,
    //GDF_PRODUCT,
    //GDF_DOT
};

/* additonal flags */
typedef struct gdf_context_{
  int flag_sorted;        /* 0 = No, 1 = yes */
  gdf_method flag_method; /* what method is used */
  int flag_distinct;      /* for COUNT: DISTINCT = 1, else = 0 */
} gdf_context;

struct _OpaqueIpcParser;
typedef struct _OpaqueIpcParser gdf_ipc_parser_type;


struct _OpaqueRadixsortPlan;
typedef struct _OpaqueRadixsortPlan gdf_radixsort_plan_type;


struct _OpaqueSegmentedRadixsortPlan;
typedef struct _OpaqueSegmentedRadixsortPlan gdf_segmented_radixsort_plan_type;


struct _OpaqueJoinResult;
typedef struct _OpaqueJoinResult gdf_join_result_type;
