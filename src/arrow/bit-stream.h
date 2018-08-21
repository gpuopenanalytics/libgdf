/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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
#ifndef GDF_ARROW_UTIL_BIT_STREAM_UTILS_H
#define GDF_ARROW_UTIL_BIT_STREAM_UTILS_H

#include <algorithm>
#include <cstdint>
#include <string.h>

#include "arrow/util/bit-util.h"
#include "arrow/util/bpacking.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"

namespace gdf {
namespace arrow {
    namespace internal {

        /// Utility class to read bit/byte stream.  This class can read bits or bytes
        /// that are either byte aligned or not.  It also has utilities to read multiple
        /// bytes in one read (e.g. encoded int).
        class BitReader {
        public:
            /// 'buffer' is the buffer to read from.  The buffer's length is 'buffer_len'.
            BitReader(const uint8_t* buffer, int buffer_len)
                : buffer_(buffer)
                , max_bytes_(buffer_len)
                , byte_offset_(0)
                , bit_offset_(0)
            {
                int num_bytes = std::min(8, max_bytes_ - byte_offset_);
                memcpy(&buffered_values_, buffer_ + byte_offset_, num_bytes);
            }

            BitReader()
                : buffer_(NULL)
                , max_bytes_(0)
            {
            }

            void Reset(const uint8_t* buffer, int buffer_len)
            {
                buffer_ = buffer;
                max_bytes_ = buffer_len;
                byte_offset_ = 0;
                bit_offset_ = 0;
                int num_bytes = std::min(8, max_bytes_ - byte_offset_);
                memcpy(&buffered_values_, buffer_ + byte_offset_, num_bytes);
            }

            /// Gets the next value from the buffer.  Returns true if 'v' could be read or
            /// false if
            /// there are not enough bytes left. num_bits must be <= 32.
            template <typename T>
            bool GetValue(int num_bits, T* v);

            template <typename T>
            void SetGpuBatchMetadata(int num_bits, T* v, int batch_size, int values_read,  
                std::vector<int>& unpack32InputOffsets,
                std::vector< std::pair<uint32_t, uint32_t> >& bitpackset,
                std::vector<int>& unpack32OutputOffsets,
                std::vector<int>& remainderInputOffsets,
                std::vector<int>& remainderBitOffsets,
                std::vector<int>& remainderSetSize,
                std::vector<int>& remainderOutputOffsets);

            /// Get a number of values from the buffer. Return the number of values
            /// actually read.
            template <typename T>
            int GetBatch(int num_bits, T* v, int batch_size);

            /// Reads a 'num_bytes'-sized value from the buffer and stores it in 'v'. T
            /// needs to be a little-endian native type and big enough to store
            /// 'num_bytes'. The value is assumed to be byte-aligned so the stream will
            /// be advanced to the start of the next byte before 'v' is read. Returns
            /// false if there are not enough bytes left.
            template <typename T>
            bool GetAligned(int num_bytes, T* v);

            /// Reads a vlq encoded int from the stream.  The encoded int must start at
            /// the beginning of a byte. Return false if there were not enough bytes in
            /// the buffer.
            bool GetVlqInt(int32_t* v);

            // Reads a zigzag encoded int `into` v.
            bool GetZigZagVlqInt(int32_t* v);

            /// Returns the number of bytes left in the stream, not including the current
            /// byte (i.e., there may be an additional fraction of a byte).
            int bytes_left()
            {
                return max_bytes_ - (byte_offset_ + static_cast<int>(::arrow::BitUtil::Ceil(
                                                        bit_offset_, 8)));
            }

            const uint8_t* get_buffer() { return buffer_; }
            int get_buffer_len() { return max_bytes_; }
            /// Maximum byte length of a vlq encoded int
            static const int MAX_VLQ_BYTE_LEN = 5;

        private:
            const uint8_t* buffer_;
            int max_bytes_;

            /// Bytes are memcpy'd from buffer_ and values are read from this variable.
            /// This is
            /// faster than reading values byte by byte directly from buffer_.
            uint64_t buffered_values_;

            int byte_offset_; // Offset in buffer_
            int bit_offset_; // Offset in buffered_values_
        };



        template <typename T>
        inline void GetValue_(int num_bits, T* v, int max_bytes, const uint8_t* buffer,
                            int* bit_offset, int* byte_offset, uint64_t* buffered_values) {
            #ifdef _MSC_VER
            #pragma warning(push)
            #pragma warning(disable : 4800)
            #endif
            *v = static_cast<T>(::arrow::BitUtil::TrailingBits(*buffered_values, *bit_offset + num_bits) >>
                                *bit_offset);
            #ifdef _MSC_VER
            #pragma warning(pop)
            #endif
            *bit_offset += num_bits;
            if (*bit_offset >= 64) {
                *byte_offset += 8;
                *bit_offset -= 64;

                int bytes_remaining = max_bytes - *byte_offset;
                if (ARROW_PREDICT_TRUE(bytes_remaining >= 8)) {
                memcpy(buffered_values, buffer + *byte_offset, 8);
                } else {
                memcpy(buffered_values, buffer + *byte_offset, bytes_remaining);
                }
            #ifdef _MSC_VER
            #pragma warning(push)
            #pragma warning(disable : 4800 4805)
            #endif
                // Read bits of v that crossed into new buffered_values_
                *v = *v | static_cast<T>(::arrow::BitUtil::TrailingBits(*buffered_values, *bit_offset)
                                        << (num_bits - *bit_offset));
            #ifdef _MSC_VER
            #pragma warning(pop)
            #endif
                DCHECK_LE(*bit_offset, 64);
            }
        }

        template <typename T>
        inline bool BitReader::GetValue(int num_bits, T* v)
        {
            return GetBatch(num_bits, v, 1) == 1;
        }


        template <typename T>
        inline void
        BitReader::SetGpuBatchMetadata(int num_bits, T* v, int batch_size, int values_read,
            std::vector<int>& unpack32InputOffsets,
            std::vector< std::pair<uint32_t, uint32_t> > &bitpackset,
            std::vector<int>& unpack32OutputOffsets,
            std::vector<int>& remainderInputOffsets,
            std::vector<int>& remainderBitOffsets,
            std::vector<int>& remainderSetSize,
            std::vector<int>& remainderOutputOffsets)
        {
            DCHECK(buffer_ != NULL);
            // TODO: revisit this limit if necessary
            DCHECK_LE(num_bits, 32);
            //	  DCHECK_LE(num_bits, static_cast<int>(sizeof(T) * 8));

        
            int bit_offset = bit_offset_;
            int byte_offset = byte_offset_;
            uint64_t buffered_values = buffered_values_;
            int max_bytes = max_bytes_;
            const uint8_t* buffer = buffer_;

            uint64_t needed_bits = num_bits * batch_size;
            uint64_t remaining_bits = (max_bytes - byte_offset) * 8 - bit_offset;
            if (remaining_bits < needed_bits) {
                batch_size = static_cast<int>(remaining_bits) / num_bits;
            }

            int i = 0;
            if (ARROW_PREDICT_FALSE(bit_offset != 0)) {
                int byte_offset_start = byte_offset;
                int bit_offset_start = bit_offset;
                int i_start = i + values_read;

                int count = 0;
                for (; i < batch_size && bit_offset != 0; ++i) {
                    bit_offset += num_bits;
                    if (bit_offset >= 64) {
                        byte_offset += 8;
                        bit_offset -= 64;
                    }
                    count++;
                }
                if (count > 0) {
                    remainderInputOffsets.push_back(byte_offset_start);
                    remainderBitOffsets.push_back(bit_offset_start);
                    remainderOutputOffsets.push_back(i_start);
                    remainderSetSize.push_back(count);
                }
            }

            int unpack_batch_size = (batch_size - i) / 32 * 32;
            int num_loops = unpack_batch_size / 32;
            int start_input_offset = byte_offset;
            for (int j = 0; j < num_loops; ++j) {
                unpack32InputOffsets.push_back(byte_offset);
                unpack32OutputOffsets.push_back(i + values_read);
                i += 32;
                byte_offset += 32 * num_bits / 8;
                
            }
            if (num_loops > 0) {
                bitpackset.push_back(std::make_pair<uint32_t, uint32_t>(start_input_offset, byte_offset - start_input_offset));
            }
            int byte_offset_start = byte_offset;
            int bit_offset_start = bit_offset;
            int i_start = i + values_read;

            int count = 0;
            for (; i < batch_size; ++i) {
                bit_offset += num_bits;
                if (bit_offset >= 64) {
                    byte_offset += 8;
                    bit_offset -= 64;
                }
                count++;
            }
            if (count > 0) {
                remainderInputOffsets.push_back(byte_offset_start);
                remainderBitOffsets.push_back(bit_offset_start);
                remainderOutputOffsets.push_back(i_start);
                remainderSetSize.push_back(count);
            }
            
            bit_offset_ = bit_offset;
            byte_offset_ = byte_offset;
            buffered_values_ = buffered_values;
        }

        template <typename T>
        inline int BitReader::GetBatch(int num_bits, T* v, int batch_size)
        {
            DCHECK(buffer_ != NULL);
            // TODO: revisit this limit if necessary
            DCHECK_LE(num_bits, 32);
            DCHECK_LE(num_bits, static_cast<int>(sizeof(T) * 8));

            int bit_offset = bit_offset_;
            int byte_offset = byte_offset_;
            uint64_t buffered_values = buffered_values_;
            int max_bytes = max_bytes_;
            const uint8_t* buffer = buffer_;

            uint64_t needed_bits = num_bits * batch_size;
            uint64_t remaining_bits = (max_bytes - byte_offset) * 8 - bit_offset;
            if (remaining_bits < needed_bits) {
                batch_size = static_cast<int>(remaining_bits) / num_bits;
            }

            int i = 0;
            if (ARROW_PREDICT_FALSE(bit_offset != 0)) {
                for (; i < batch_size && bit_offset != 0; ++i) {
                    GetValue_(num_bits, &v[i], max_bytes, buffer,
                        &bit_offset, &byte_offset, &buffered_values);
                }
            }

            if (sizeof(T) == 4) {
                int num_unpacked = ::arrow::internal::unpack32(
                    reinterpret_cast<const uint32_t*>(buffer + byte_offset),
                    reinterpret_cast<uint32_t*>(v + i), batch_size - i, num_bits);
                i += num_unpacked;
                byte_offset += num_unpacked * num_bits / 8;
            } else {
                const int buffer_size = 1024;
                uint32_t unpack_buffer[buffer_size];
                while (i < batch_size) {
                    int unpack_size = std::min(buffer_size, batch_size - i);
                    int num_unpacked = ::arrow::internal::unpack32(
                        reinterpret_cast<const uint32_t*>(buffer + byte_offset),
                        unpack_buffer, unpack_size, num_bits);
                    if (num_unpacked == 0) {
                        break;
                    }
                    for (int k = 0; k < num_unpacked; ++k) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif
                        v[i + k] = static_cast<T>(unpack_buffer[k]);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
                    }
                    i += num_unpacked;
                    byte_offset += num_unpacked * num_bits / 8;
                }
            }

            int bytes_remaining = max_bytes - byte_offset;
            if (bytes_remaining >= 8) {
                memcpy(&buffered_values, buffer + byte_offset, 8);
            } else {
                memcpy(&buffered_values, buffer + byte_offset, bytes_remaining);
            }
            for (; i < batch_size; ++i) {
                GetValue_(num_bits, &v[i], max_bytes, buffer, &bit_offset,
                    &byte_offset, &buffered_values);
            }
            bit_offset_ = bit_offset;
            byte_offset_ = byte_offset;
            buffered_values_ = buffered_values;

            return batch_size;
        }

        template <typename T>
        inline bool BitReader::GetAligned(int num_bytes, T* v)
        {
            DCHECK_LE(num_bytes, static_cast<int>(sizeof(T)));
            int bytes_read = static_cast<int>(::arrow::BitUtil::Ceil(bit_offset_, 8));
            if (ARROW_PREDICT_FALSE(byte_offset_ + bytes_read + num_bytes > max_bytes_))
                return false;

            // Advance byte_offset to next unread byte and read num_bytes
            byte_offset_ += bytes_read;
            memcpy(v, buffer_ + byte_offset_, num_bytes);
            byte_offset_ += num_bytes;

            // Reset buffered_values_
            bit_offset_ = 0;
            int bytes_remaining = max_bytes_ - byte_offset_;
            if (ARROW_PREDICT_TRUE(bytes_remaining >= 8)) {
                memcpy(&buffered_values_, buffer_ + byte_offset_, 8);
            } else {
                memcpy(&buffered_values_, buffer_ + byte_offset_, bytes_remaining);
            }
            return true;
        }

        inline bool BitReader::GetVlqInt(int32_t* v)
        {
            *v = 0;
            int shift = 0;
            int num_bytes = 0;
            uint8_t byte = 0;
            do {
                if (!GetAligned<uint8_t>(1, &byte))
                    return false;
                *v |= (byte & 0x7F) << shift;
                shift += 7;
                DCHECK_LE(++num_bytes, MAX_VLQ_BYTE_LEN);
            } while ((byte & 0x80) != 0);
            return true;
        }

        inline bool BitReader::GetZigZagVlqInt(int32_t* v)
        {
            int32_t u_signed;
            if (!GetVlqInt(&u_signed))
                return false;
            uint32_t u = static_cast<uint32_t>(u_signed);
            *reinterpret_cast<uint32_t*>(v) = (u >> 1) ^ -(static_cast<int32_t>(u & 1));
            return true;
        }
    }
}
}

#endif