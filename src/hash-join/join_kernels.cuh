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

constexpr int JoinNoneValue = -1;

enum class JoinType {
  INNER_JOIN,
  LEFT_JOIN,
};

#include "../gdf_table.cuh"
#include "concurrent_unordered_multimap.cuh"
#include <cub/cub.cuh>

constexpr int warp_size = 32;

template<typename multimap_type,
         typename key_type = typename multimap_type::key_type,
         typename size_type = typename multimap_type::size_type>
__global__ void build_hash_table( multimap_type * const multi_map,
                                  const key_type * const build_column,
                                  const size_type build_column_size)
{
    const size_type i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < build_column_size ) {
      multi_map->insert( thrust::make_pair( build_column[i], i ) );
    }
}

template<typename size_type,
         typename join_output_pair>
__inline__ __device__ void add_pair_to_cache(const size_type first, 
                                             const size_type second, 
                                             int *current_idx_shared, 
                                             const int warp_id, 
                                             join_output_pair *joined_shared)
{
  join_output_pair joined_val;
  joined_val.first = first;
  joined_val.second = second;

  int my_current_idx = atomicAdd(current_idx_shared + warp_id, 1);

  // its guaranteed to fit into the shared cache
  joined_shared[my_current_idx] = joined_val;
}

template< JoinType join_type,
          typename multimap_type,
          typename key_type,
          typename size_type,
          int block_size,
          int output_cache_size>
__global__ void compute_join_output_size( multimap_type const * const multi_map,
                                          gdf_table<size_type> const & build_table,
                                          gdf_table<size_type> const & probe_table,
                                          key_type const * const probe_column,
                                          const size_type probe_table_size,
                                          size_type* output_size)
{

  __shared__ size_type block_counter;
  block_counter=0;
  __syncthreads();

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  __syncwarp();
#endif

  size_type probe_row_index = threadIdx.x + blockIdx.x * blockDim.x;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  const unsigned int activemask = __ballot_sync(0xffffffff, probe_row_index < probe_table_size);
#endif
  if ( probe_row_index < probe_table_size ) {
    const auto unused_key = multi_map->get_unused_key();
    const auto end = multi_map->end();
    const key_type probe_key = probe_column[probe_row_index];
    auto found = multi_map->find(probe_key);

    bool running = (join_type == JoinType::LEFT_JOIN) || (end != found); // for left-joins we always need to add an output
    bool found_match = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    while ( __any_sync( activemask, running ) )
#else
      while ( __any( running ) )
#endif
      {
        if ( running )
        {
          if (join_type == JoinType::LEFT_JOIN && (end == found)) {
            running = false;    // add once on the first iteration
          }
          else if ( unused_key == found->first ) {
            running = false;
          }
          else if (false == probe_table.rows_equal(build_table, probe_row_index, found->second))
          {

            // Continue searching for matching rows until you hit an empty hash table entry
            ++found;
            if(end == found)
              found = multi_map->begin();

            if(unused_key == found->first)
              running = false;
          }
          else 
          {
            found_match = true;

            atomicAdd(&block_counter,static_cast<size_type>(1)) ;

            // Continue searching for matching rows until you hit an empty hash table entry
            ++found;
            if(end == found)
              found = multi_map->begin();

            if(unused_key == found->first)
              running = false;
          }

          if ((join_type == JoinType::LEFT_JOIN) && (!running) && (!found_match)) {
            atomicAdd(&block_counter,static_cast<size_type>(1));
          }
        }
      }
  }

  __syncthreads();

  // Add block counter to global counter
  if (threadIdx.x==0)
    atomicAdd(output_size, block_counter);
}


template< JoinType join_type,
          typename multimap_type,
          typename key_type,
          typename size_type,
          typename join_output_pair,
          size_type block_size,
          size_type output_cache_size>
__global__ void probe_hash_table( multimap_type const * const multi_map,
                                  gdf_table<size_type> const & build_table,
                                  gdf_table<size_type> const & probe_table,
                                  key_type const * const probe_column,
                                  const size_type probe_table_size,
                                  join_output_pair * const join_output,
                                  size_type* current_idx,
                                  const size_type max_size,
                                  const size_type offset = 0)
{
  constexpr int num_warps = block_size/warp_size;
  __shared__ int current_idx_shared[num_warps];
  __shared__ join_output_pair join_output_shared[num_warps][output_cache_size];

  const int warp_id = threadIdx.x/warp_size;
  const int lane_id = threadIdx.x%warp_size;

  if ( 0 == lane_id )
  {
    current_idx_shared[warp_id] = 0;
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  __syncwarp();
#endif

  size_type probe_row_index = threadIdx.x + blockIdx.x * blockDim.x;


#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  const unsigned int activemask = __ballot_sync(0xffffffff, probe_row_index < probe_table_size);
#endif
  if ( probe_row_index < probe_table_size ) {
    const auto unused_key = multi_map->get_unused_key();
    const auto end = multi_map->end();
    const key_type probe_key = probe_column[probe_row_index];
    auto found = multi_map->find(probe_key);

    bool running = (join_type == JoinType::LEFT_JOIN) || (end != found);	// for left-joins we always need to add an output
    bool found_match = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    while ( __any_sync( activemask, running ) )
#else
      while ( __any( running ) )
#endif
      {
        if ( running )
        {
          if (join_type == JoinType::LEFT_JOIN && (end == found)) {
            // add once on the first iteration
            running = false;	
          }
          else if ( unused_key == found->first ) {
            running = false;
          }
          else if ( false == probe_table.rows_equal(build_table, probe_row_index, found->second) ){

            // Keep searching for matches until you encounter an empty hash table location 
            ++found;
            if(end == found)
              found = multi_map->begin();

            if(unused_key == found->first)
              running = false;
          }
          else {
            found_match = true;

            add_pair_to_cache(offset + probe_row_index, found->second, current_idx_shared, warp_id, join_output_shared[warp_id]);

            // Keep searching for matches until you encounter an empty hash table location 
            ++found;
            if(end == found)
              found = multi_map->begin();

            if(unused_key == found->first)
              running = false;
          }
          if ((join_type == JoinType::LEFT_JOIN) && (!running) && (!found_match)) {
            add_pair_to_cache(offset + probe_row_index, JoinNoneValue, current_idx_shared, warp_id, join_output_shared[warp_id]);
          }
        }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
        __syncwarp(activemask);
#endif
        //flush output cache if next iteration does not fit
        if ( current_idx_shared[warp_id] + warp_size >= output_cache_size ) {

          // count how many active threads participating here which could be less than warp_size
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
          const unsigned int activemask = __ballot(1);
#endif
          int num_threads = __popc(activemask);
          unsigned long long int output_offset = 0;

          if ( 0 == lane_id )
          {
            output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
          }

          output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

          for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) 
          {
            size_type thread_offset = output_offset + shared_out_idx;
            if (thread_offset < max_size)
              join_output[thread_offset] = join_output_shared[warp_id][shared_out_idx];
          }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
          __syncwarp(activemask);
#endif
          if ( 0 == lane_id )
            current_idx_shared[warp_id] = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
          __syncwarp(activemask);
#endif
        }
      }

    //final flush of output cache
    if ( current_idx_shared[warp_id] > 0 ) 
    {
      // count how many active threads participating here which could be less than warp_size
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
      const unsigned int activemask = __ballot(1);
#endif
      int num_threads = __popc(activemask);
      unsigned long long int output_offset = 0;
      if ( 0 == lane_id )
      {
        output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
      }
        
      output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

      for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) {
        size_type thread_offset = output_offset + shared_out_idx;
        if (thread_offset < max_size)
          join_output[thread_offset] = join_output_shared[warp_id][shared_out_idx];
      }
    }
  }
}

/*
template<
    typename multimap_type,
    typename key_type,
    typename size_type,
    typename join_output_pair,
    int block_size>
__global__ void probe_hash_table_uniq_keys(
    multimap_type * multi_map,
    const key_type* probe_table,
    const size_type probe_table_size,
    join_output_pair * const joined,
    size_type* const current_idx,
    const size_type offset)
{
    __shared__ int current_idx_shared;
    __shared__ size_type output_offset_shared;
    __shared__ join_output_pair joined_shared[block_size];
    if ( 0 == threadIdx.x ) {
        output_offset_shared = 0;
        current_idx_shared = 0;
    }
    
    __syncthreads();

    size_type i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < probe_table_size ) {
        const auto end = multi_map->end();
        auto found = multi_map->find(probe_table[i]);
        if ( end != found ) {
            join_output_pair joined_val;
            joined_val.first = offset+i;
            joined_val.second = found->second;
            int my_current_idx = atomicAdd( &current_idx_shared, 1 );
            //its guranteed to fit into the shared cache
            joined_shared[my_current_idx] = joined_val;
        }
    }
    
    __syncthreads();
    
    if ( current_idx_shared > 0 ) {
        if ( 0 == threadIdx.x ) {
            output_offset_shared = atomicAdd( current_idx, current_idx_shared );
        }
        __syncthreads();
        
        if ( threadIdx.x < current_idx_shared ) {
            joined[output_offset_shared+threadIdx.x] = joined_shared[threadIdx.x];
        }
    }
}

*/
