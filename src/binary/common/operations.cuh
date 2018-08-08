#ifndef GDF_BINARY_COMMON_OPERATIONS_CUH
#define GDF_BINARY_COMMON_OPERATIONS_CUH

#include <cstdint>
#include <thrust/device_vector.h>

namespace gdf {
namespace binary {
namespace common {

    namespace {
        template <typename Type>
        Type getFromUnion(gdf_scalar::gdf_data);

        template <>
        int8_t getFromUnion<int8_t>(gdf_scalar::gdf_data data) {
            return data.si08;
        }

        template <>
        int16_t getFromUnion<int16_t>(gdf_scalar::gdf_data data) {
            return data.si16;
        }

        template <>
        int32_t getFromUnion<int32_t>(gdf_scalar::gdf_data data) {
            return data.si32;
        }

        template <>
        int64_t getFromUnion<int64_t>(gdf_scalar::gdf_data data) {
            return data.si64;
        }

        template <>
        float getFromUnion<float>(gdf_scalar::gdf_data data) {
            return data.fp32;
        }

        template <>
        double getFromUnion<double>(gdf_scalar::gdf_data data) {
            return data.fp64;
        }
    }


    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    gdf_error operate(gdf_column* out, gdf_column* vax, GdfObject* vay, gdf_scalar* def, TypeOpe&& ope) {

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        thrust::device_ptr<TypeOut> out_ptr((TypeOut*) out->data);
        thrust::device_ptr<TypeVax> vax_ptr((TypeVax*) vax->data);
        //thrust::device_ptr<TypeVay> vay_ptr;//((TypeVay*) vay->data);

        if (vay->getType() == GdfType::Scalar) {
            auto constant_iterator = thrust::make_constant_iterator<TypeVay>(getFromUnion<TypeVay>(vay->getScalar()->data));
            thrust::transform(
                    thrust::cuda::par.on(stream),
                    thrust::detail::make_normal_iterator(vax_ptr),
                    thrust::detail::make_normal_iterator(vax_ptr) + vax->size,
                    constant_iterator,
                    thrust::detail::make_normal_iterator(out_ptr),
                    ope);
        } else {
            thrust::device_ptr<TypeVay> vay_ptr((TypeVay*) vay->getVector()->data);
            thrust::transform(
                    thrust::cuda::par.on(stream),
                    thrust::detail::make_normal_iterator(vax_ptr),
                    thrust::detail::make_normal_iterator(vax_ptr) + vax->size,
                    thrust::detail::make_normal_iterator(vay_ptr),
                    thrust::detail::make_normal_iterator(out_ptr),
                    ope);
        }

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        return GDF_SUCCESS;
    }

}
}
}

#endif
