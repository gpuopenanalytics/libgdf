#ifndef _CU_DECODER_H_
#define _CU_DECODER_H_

namespace gdf
{
namespace arrow
{

int decode_using_cpu(const uint8_t *buffer,
                     const int buffer_len,
                     const std::vector<uint32_t> &rle_runs,
                     const std::vector<uint64_t> &rle_values,
                     const std::vector<int> &input_offset,
                     const std::vector<int> &output_offset,
                     const std::vector<int> &remainderInputOffsets,
                     const std::vector<int> &remainderBitOffsets,
                     const std::vector<int> &remainderSetSize,
                     const std::vector<int> &remainderOutputOffsets,
                     const std::vector<uint16_t> &is_rle,
                     int num_bits,
                     int *output,
                     int batch_size);

int decode_using_gpu(const uint8_t *buffer,
                     const int buffer_len,
                     const std::vector<uint32_t> &rle_runs,
                     const std::vector<uint64_t> &rle_values,
                     const std::vector<int> &input_offset,
                     const std::vector<int> &output_offset,
                     const std::vector<int> &remainderInputOffsets,
                     const std::vector<int> &remainderBitOffsets,
                     const std::vector<int> &remainderSetSize,
                     const std::vector<int> &remainderOutputOffsets,
                     const std::vector<uint16_t> &is_rle,
                     int num_bits,
                     int *output,
                     int batch_size);

} // namespace arrow
} // namespace gdf

#endif // _CU_DECODER_H_
