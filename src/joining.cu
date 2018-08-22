/*
 * Copyright (c) 2017, NVIDIA CORPORATION.
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

#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <limits>

#include "joining.h"
#include "gdf_table.cuh"

using namespace mgpu;

template <typename T>
void dump_mem(const char name[], const mem_t<T> & mem) {

    auto data = from_mem(mem);
    std::cout << name << " = " ;
    for (int i=0; i < data.size(); ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << "\n";
}

gdf_join_result_type* cffi_wrap(join_result_base *obj) {
    return reinterpret_cast<gdf_join_result_type*>(obj);
}

join_result_base* cffi_unwrap(gdf_join_result_type* hdl) {
    return reinterpret_cast<join_result_base*>(hdl);
}

gdf_error gdf_join_result_free(gdf_join_result_type *result) {
    delete cffi_unwrap(result);
    CUDA_CHECK_LAST();
    return GDF_SUCCESS;
}

void* gdf_join_result_data(gdf_join_result_type *result) {
    return cffi_unwrap(result)->data();
}

size_t gdf_join_result_size(gdf_join_result_type *result) {
    return cffi_unwrap(result)->size();
}


// Size limit due to use of int32 as join output.
// FIXME: upgrade to 64-bit
using output_type = int;
constexpr output_type MAX_JOIN_SIZE{std::numeric_limits<output_type>::max()};

#define DEF_JOIN(Fn, T, Joiner)                                             \
gdf_error gdf_##Fn(gdf_column *leftcol, gdf_column *rightcol,               \
                   gdf_join_result_type **out_result) {                     \
    using namespace mgpu;                                                   \
    if ( leftcol->dtype != rightcol->dtype) return GDF_UNSUPPORTED_DTYPE;   \
    if ( leftcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;   \
    if ( rightcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;  \
    std::unique_ptr<join_result<int> > result_ptr(new join_result<int>);    \
    result_ptr->result = Joiner((T*)leftcol->data, leftcol->size,           \
                                (T*)rightcol->data, rightcol->size,         \
                                less_t<T>(), result_ptr->context);          \
    CUDA_CHECK_LAST();                                                      \
    *out_result = cffi_wrap(result_ptr.release());                          \
    return GDF_SUCCESS;                                                     \
}

#define DEF_JOIN_GENERIC(Fn)                                               \
gdf_error gdf_##Fn##_generic(gdf_column *leftcol, gdf_column * rightcol,   \
                                 gdf_join_result_type **out_result) {      \
    switch ( leftcol->dtype ){                                             \
    case GDF_INT8:    return gdf_##Fn##_i8 (leftcol, rightcol, out_result);\
    case GDF_INT16:   return gdf_##Fn##_i16(leftcol, rightcol, out_result);\
    case GDF_INT32:   return gdf_##Fn##_i32(leftcol, rightcol, out_result);\
    case GDF_INT64:   return gdf_##Fn##_i64(leftcol, rightcol, out_result);\
    case GDF_FLOAT32: return gdf_##Fn##_f32(leftcol, rightcol, out_result);\
    case GDF_FLOAT64: return gdf_##Fn##_f64(leftcol, rightcol, out_result);\
    default: return GDF_UNSUPPORTED_DTYPE;                                 \
    }                                                                      \
}

#define DEF_OUTER_JOIN(Fn, T) DEF_JOIN(outer_join_ ## Fn, T, outer_join)
DEF_JOIN_GENERIC(outer_join)
DEF_OUTER_JOIN(i8,  int8_t)
DEF_OUTER_JOIN(i16, int16_t)
DEF_OUTER_JOIN(i32, int32_t)
DEF_OUTER_JOIN(i64, int64_t)
DEF_OUTER_JOIN(f32, int32_t)
DEF_OUTER_JOIN(f64, int64_t)

#define JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, T3, l3, r3) \
  result_ptr->result = join_hash<join_type>( \
                                (T1*)l1, (int)leftcol[0]->size, \
                                (T1*)r1, (int)rightcol[0]->size, \
                                (T2*)l2, (T2*)r2, \
                                (T3*)l3, (T3*)r3, \
                                less_t<int64_t>(), result_ptr->context);

#define JOIN_HASH_T3(T1, l1, r1, T2, l2, r2, T3, l3, r3) \
  if (T3 == GDF_INT8)      { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2,  int8_t, l3, r3) } \
  if (T3 == GDF_INT16)     { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int16_t, l3, r3) } \
  if (T3 == GDF_INT32)     { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int32_t, l3, r3) } \
  if (T3 == GDF_INT64)     { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int64_t, l3, r3) } \
  if (T3 == GDF_FLOAT32)   { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int32_t, l3, r3) } \
  if (T3 == GDF_FLOAT64)   { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int64_t, l3, r3) } \
  if (T3 == GDF_DATE32)    { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int32_t, l3, r3) } \
  if (T3 == GDF_DATE64)    { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int64_t, l3, r3) } \
  if (T3 == GDF_TIMESTAMP) { JOIN_HASH_TYPES(T1, l1, r1, T2, l2, r2, int64_t, l3, r3) }

#define JOIN_HASH_T2(T1, l1, r1, T2, l2, r2, T3, l3, r3) \
  if (T2 == GDF_INT8)       { JOIN_HASH_T3(T1, l1, r1,  int8_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_INT16)      { JOIN_HASH_T3(T1, l1, r1, int16_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_INT32)      { JOIN_HASH_T3(T1, l1, r1, int32_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_INT64)      { JOIN_HASH_T3(T1, l1, r1, int64_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_FLOAT32)    { JOIN_HASH_T3(T1, l1, r1, int32_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_FLOAT64)    { JOIN_HASH_T3(T1, l1, r1, int64_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_DATE32)     { JOIN_HASH_T3(T1, l1, r1, int32_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_DATE64)     { JOIN_HASH_T3(T1, l1, r1, int64_t, l2, r2, T3, l3, r3) } \
  if (T2 == GDF_TIMESTAMP)  { JOIN_HASH_T3(T1, l1, r1, int64_t, l2, r2, T3, l3, r3) }

#define JOIN_HASH_T1(T1, l1, r1, T2, l2, r2, T3, l3, r3) \
  if (T1 == GDF_INT8)      { JOIN_HASH_T2( int8_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_INT16)     { JOIN_HASH_T2(int16_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_INT32)     { JOIN_HASH_T2(int32_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_INT64)     { JOIN_HASH_T2(int64_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_FLOAT32)   { JOIN_HASH_T2(int32_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_FLOAT64)   { JOIN_HASH_T2(int64_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_DATE32)    { JOIN_HASH_T2(int32_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_DATE64)    { JOIN_HASH_T2(int64_t, l1, r1, T2, l2, r2, T3, l3, r3) } \
  if (T1 == GDF_TIMESTAMP) { JOIN_HASH_T2(int64_t, l1, r1, T2, l2, r2, T3, l3, r3) }

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Computes the Join result between two tables using the hash-based implementation. 
 * 
 * @Param num_cols The number of columns to join
 * @Param leftcol The left set of columns to join
 * @Param rightcol The right set of columns to join
 * @Param out_result The result of the join operation. The first n/2 elements of the
   output are the left indices, the last n/2 elements of the output are the right indices.
   @tparam join_type The type of join to be performed
 * 
 * @Returns Upon successful computation, returns GDF_SUCCESS. Otherwise returns appropriate error code 
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type, 
          typename size_type>
gdf_error hash_join(size_type num_cols, gdf_column **leftcol, gdf_column **rightcol, gdf_join_result_type **out_result)
{
  // Wrap the set of gdf_columns in a gdf_table class
  std::unique_ptr< gdf_table<size_type> > left_table(new gdf_table<size_type>(num_cols, leftcol));
  std::unique_ptr< gdf_table<size_type> > right_table(new gdf_table<size_type>(num_cols, rightcol));

  std::unique_ptr< join_result<output_type> > result_ptr(new join_result<output_type>);

  result_ptr->result = join_hash<join_type, output_type>(*left_table, *right_table, result_ptr->context);

  CUDA_CHECK_LAST();
  *out_result = cffi_wrap(result_ptr.release());
  return GDF_SUCCESS;
}

template <JoinType join_type>
struct SortJoin {
template<typename launch_arg_t = mgpu::empty_t,
  typename a_it, typename b_it, typename comp_t>
    mgpu::mem_t<int> operator()(a_it a, int a_count, b_it b, int b_count,
                                 comp_t comp, context_t& context) {
        return mem_t<int>();
    }
};

template <>
struct SortJoin<JoinType::INNER_JOIN> {
template<typename launch_arg_t = mgpu::empty_t,
  typename a_it, typename b_it, typename comp_t>
    mgpu::mem_t<int> operator()(a_it a, int a_count, b_it b, int b_count,
                                 comp_t comp, context_t& context) {
        return inner_join(a, a_count, b, b_count, comp, context);
    }
};

template <>
struct SortJoin<JoinType::LEFT_JOIN> {
  template<typename launch_arg_t = mgpu::empty_t,
    typename a_it, typename b_it, typename comp_t>
      mgpu::mem_t<int> operator()(a_it a, int a_count, b_it b, int b_count,
                                  comp_t comp, context_t& context) {
        return left_join(a, a_count, b, b_count, comp, context);
      }
};

template <JoinType join_type, typename T>
gdf_error sort_join_typed(gdf_column *leftcol, gdf_column *rightcol,
                          gdf_join_result_type **out_result, gdf_context *ctxt) 
{
  using namespace mgpu;
  gdf_error err = GDF_SUCCESS;

  std::unique_ptr<join_result<int> > result_ptr(new join_result<int>);

  SortJoin<join_type> sort_based_join;
  result_ptr->result = sort_based_join(static_cast<T*>(leftcol->data), leftcol->size,
                                       static_cast<T*>(rightcol->data), rightcol->size,
                                       less_t<T>(), result_ptr->context);
  CUDA_CHECK_LAST();
  *out_result = cffi_wrap(result_ptr.release());

  return err;
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Computes the join operation between a single left and single right column
 using the sort based implementation.
 * 
 * @Param leftcol The left column to join
 * @Param rightcol The right column to join
 * @Param out_result The output of the join operation
 * @Param ctxt Structure that determines various run parameters, such as if the inputs
 are already sorted.
   @tparama join_type The type of join to perform
 * 
 * @Returns GDF_SUCCESS upon succesful completion of the join, otherwise returns 
 appropriate error code.
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
gdf_error sort_join(gdf_column *leftcol, gdf_column *rightcol,
                    gdf_join_result_type **out_result, gdf_context *ctxt) 
{

  if(GDF_SORT != ctxt->flag_method) return GDF_INVALID_API_CALL;

  switch ( leftcol->dtype ){
    case GDF_INT8:    return sort_join_typed<join_type, int8_t>(leftcol, rightcol, out_result, ctxt);
    case GDF_INT16:   return sort_join_typed<join_type,int16_t>(leftcol, rightcol, out_result, ctxt);
    case GDF_INT32:   return sort_join_typed<join_type,int32_t>(leftcol, rightcol, out_result, ctxt);
    case GDF_INT64:   return sort_join_typed<join_type,int64_t>(leftcol, rightcol, out_result, ctxt);
    case GDF_FLOAT32: return sort_join_typed<join_type,int32_t>(leftcol, rightcol, out_result, ctxt);
    case GDF_FLOAT64: return sort_join_typed<join_type,int64_t>(leftcol, rightcol, out_result, ctxt);
    default: return GDF_UNSUPPORTED_DTYPE;
  }
}

template
gdf_error sort_join<JoinType::INNER_JOIN>(gdf_column *leftcol, gdf_column *rightcol,
                             gdf_join_result_type **out_result, gdf_context *ctxt);
template
gdf_error sort_join<JoinType::LEFT_JOIN>(gdf_column *leftcol, gdf_column *rightcol,
                             gdf_join_result_type **out_result, gdf_context *ctxt);

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Computes the join operation between two sets of columns
 * 
 * @Param num_cols The number of columns to join
 * @Param leftcol The left set of columns to join
 * @Param rightcol The right set of columns to join
 * @Param out_result The result of the join operation. The output is structured such that
 * the pair (i, i + output_size/2) is the (left, right) index of matching rows.
 * @Param join_context A structure that determines various run parameters, such as
   whether to perform a hash or sort based join
 * @tparam join_type The type of join to be performed
 * 
 * @Returns GDF_SUCCESS upon succesfull compute, otherwise returns appropriate error code
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
gdf_error join_call( int num_cols, gdf_column **leftcol, gdf_column **rightcol,
                     gdf_join_result_type **out_result, gdf_context *join_context) 
{

  if( (0 == num_cols) || (nullptr == leftcol) || (nullptr == rightcol))
    return GDF_DATASET_EMPTY;

  if(nullptr == join_context)
    return GDF_INVALID_API_CALL;

  // check that the columns data are not null, have matching types, 
  // and the same number of rows
  const auto left_col_size = leftcol[0]->size;
  const auto right_col_size = rightcol[0]->size;
  
  if(left_col_size >= MAX_JOIN_SIZE) return GDF_COLUMN_SIZE_TOO_BIG;
  if(right_col_size >= MAX_JOIN_SIZE) return GDF_COLUMN_SIZE_TOO_BIG;

  for (int i = 0; i < num_cols; i++) {
    if(nullptr == rightcol[i]->data) return GDF_DATASET_EMPTY;
    if(nullptr == leftcol[i]->data) return GDF_DATASET_EMPTY;
    if(rightcol[i]->dtype != leftcol[i]->dtype) return GDF_JOIN_DTYPE_MISMATCH;
    if(left_col_size != leftcol[i]->size) return GDF_COLUMN_SIZE_MISMATCH;
    if(right_col_size != rightcol[i]->size) return GDF_COLUMN_SIZE_MISMATCH;
  }

  gdf_method join_method = join_context->flag_method; 

  switch(join_method)
  {
    case GDF_HASH:
      {
        return hash_join<join_type>(num_cols, leftcol, rightcol, out_result);
      }
    case GDF_SORT:
      {
        // Sort based joins only support single column joins
        if(1 == num_cols)
        {
          return sort_join<join_type>(leftcol[0], rightcol[0], out_result, join_context);
        }
        else
        {
          return GDF_JOIN_TOO_MANY_COLUMNS;
        }
      }
    default:
      return GDF_UNSUPPORTED_METHOD;
  }

}

gdf_error gdf_left_join(int num_cols, gdf_column **leftcol, gdf_column **rightcol,
                        gdf_join_result_type **out_result, gdf_context *ctxt) 
{
  return join_call<JoinType::LEFT_JOIN>(num_cols, leftcol, rightcol, out_result, ctxt);
}

gdf_error gdf_inner_join(int num_cols, gdf_column **leftcol, gdf_column **rightcol,
                         gdf_join_result_type **out_result, gdf_context *ctxt) 
{
  return join_call<JoinType::INNER_JOIN>(num_cols, leftcol, rightcol, out_result, ctxt);
}
