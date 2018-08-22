namespace gdf {
namespace cuda {

const char* kernel =
R"***(
    #include <cstdint>
    #include "traits.h"
    #include "operation.h"
    #include "kernel_gdf_data.h"


    #define WARP_SIZE 32


    __device__ __forceinline__
    uint32_t isValid(int tid, uint32_t* valid, uint32_t mask) {
        return valid[tid / WARP_SIZE] & mask;
    }


    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    __global__
    void kernel_v_s(int size, TypeOut* out_data, TypeVax* vax_data, gdf_data vay_data) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            AbstractOperation<TypeOpe> operation;
            out_data[tid] = operation.template operate<TypeOut, TypeVax, TypeVay>(vax_data[tid], (TypeVay)vay_data);
        }
    }


    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    __global__
    void kernel_v_v(int size, TypeOut* out_data, TypeVax* vax_data, TypeVay* vay_data) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            AbstractOperation<TypeOpe> operation;
            out_data[tid] = operation.template operate<TypeOut, TypeVax, TypeVay>(vax_data[tid], vay_data[tid]);
        }
    }


    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeDef, typename TypeOpe>
    __global__
    void kernel_v_s_d(int size, gdf_data def_data,
                      TypeOut* out_data, TypeVax* vax_data, gdf_data vay_data,
                      uint32_t* out_valid, uint32_t* vax_valid) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            uint32_t mask = 1 << (tid % WARP_SIZE);
            uint32_t is_vax_valid = isValid(tid, vax_valid, mask);

            TypeVax vax_data_aux = vax_data[tid];
            if ((is_vax_valid & mask) != mask) {
                vax_data_aux = (TypeDef)def_data;
            }

            AbstractOperation<TypeOpe> operation;
            out_data[tid] = operation.template operate<TypeOut, TypeVax, TypeVay>(vax_data_aux, (TypeVay)vay_data);
        }
    }


    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeDef, typename TypeOpe>
    __global__
    void kernel_v_v_d(int size, gdf_data def_data,
                      TypeOut* out_data, TypeVax* vax_data, TypeVax* vay_data,
                      uint32_t* out_valid, uint32_t* vax_valid, uint32_t* vay_valid) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            uint32_t mask = 1 << (tid % WARP_SIZE);
            uint32_t is_vax_valid = isValid(tid, vax_valid, mask);
            uint32_t is_vay_valid = isValid(tid, vay_valid, mask);

            TypeVax vax_data_aux = vax_data[tid];
            TypeVay vay_data_aux = vay_data[tid];
            if ((is_vax_valid & mask) != mask) {
                vax_data_aux = (TypeDef)def_data;
            }
            else if ((is_vay_valid & mask) != mask) {
                vay_data_aux = (TypeDef)def_data;
            }
            if ((is_vax_valid | is_vay_valid) == mask) {
                AbstractOperation<TypeOpe> operation;
                out_data[tid] = operation.template operate<TypeOut, TypeVax, TypeVay>(vax_data_aux, vay_data_aux);
            }
        }
    }
)***";
}
}
