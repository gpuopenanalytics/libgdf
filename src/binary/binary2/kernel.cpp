namespace gdf {
namespace cuda {

const char* kernel =
R"***(
    #include <cstdint>
    #include "traits.h"
    #include "operation.h"

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    __global__
    void kernel_v_v_s(TypeOut* out, TypeVax* vax, TypeVax vay, int size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid < size) {
            AbstractOperation<TypeOpe> operation;
            out[tid] = operation.template operate<TypeOut, TypeVax, TypeVay>(vax[tid], vay);
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    __global__
    void kernel_v_v_v(TypeOut* out, TypeVax* vax, TypeVax* vay, int size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid < size) {
            AbstractOperation<TypeOpe> operation;
            out[tid] = operation.template operate<TypeOut, TypeVax, TypeVay>(vax[tid], vay[tid]);
        }
    }

)***";
}
}
