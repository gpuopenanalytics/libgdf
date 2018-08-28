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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <limits>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <../../src/hashmap/concurrent_unordered_multimap.cuh>

// This is necessary to do a parametrized typed-test over multiple template arguments
template <typename Key, typename Value>
struct KeyValueTypes
{
  using key_type = Key;
  using value_type = Value;
};


// A new instance of this class will be created for each *TEST(MultimapTest, ...)
// Put all repeated stuff for each test here
template <class T>
class MultimapTest : public testing::Test 
{
public:
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using size_type = int;


  concurrent_unordered_multimap<key_type, 
                                value_type, 
                                size_type,
                                std::numeric_limits<key_type>::max(),
                                std::numeric_limits<value_type>::max() > the_map;

  const key_type unused_key = std::numeric_limits<key_type>::max();
  const value_type unused_value = std::numeric_limits<value_type>::max();

  const size_type size;


  MultimapTest(const size_type hash_table_size = 100)
    : the_map(hash_table_size), size(hash_table_size)
  {


  }

  ~MultimapTest(){
  }


};

// Google Test can only do a parameterized typed-test over a single type, so we have
// to nest multiple types inside of the KeyValueTypes struct above
// KeyValueTypes<type1, type2> implies key_type = type1, value_type = type2
// This list is the types across which Google Test will run our tests
typedef ::testing::Types< KeyValueTypes<int,int>, 
                          KeyValueTypes<int,long long int>,
                          KeyValueTypes<int,unsigned long long int>,
                          KeyValueTypes<unsigned long long int, int>,
                          KeyValueTypes<unsigned long long int, long long int>,
                          KeyValueTypes<unsigned long long int, unsigned long long int>
                          > Implementations;

TYPED_TEST_CASE(MultimapTest, Implementations);

TYPED_TEST(MultimapTest, InitialState)
{
  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;

  auto begin = this->the_map.begin();
  auto end = this->the_map.end();
  EXPECT_NE(begin,end);

}

TYPED_TEST(MultimapTest, CheckUnusedValues){

  EXPECT_EQ(this->the_map.get_unused_key(), this->unused_key);

  auto begin = this->the_map.begin();
  EXPECT_EQ(begin->first, this->unused_key);
  EXPECT_EQ(begin->second, this->unused_value);
}

TYPED_TEST(MultimapTest, Insert)
{
  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;

  const int NUM_PAIRS{this->size};

  // Generate a list of pairs (key, value) to insert into map
  std::vector<thrust::pair<key_type, value_type>> pairs(NUM_PAIRS);
  std::generate(pairs.begin(), pairs.end(), 
                [] () {static int i = 0; return thrust::make_pair(i,(i++)*10);});

  // Insert every pair into the map
  for(const auto& it : pairs){
    this->the_map.insert(it);
  }

  // Make sure all the pairs are in the map
  for(const auto& it : pairs){
    auto found = this->the_map.find(it.first);
    EXPECT_NE(found, this->the_map.end());
    EXPECT_EQ(found->first, it.first);
    EXPECT_EQ(found->second, it.second);
  }

}


