#ifndef GDF_TEST_LIBRARY_SCALAR_H
#define GDF_TEST_LIBRARY_SCALAR_H

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "gdf/gdf.h"
#include "library/types.h"

namespace gdf {
namespace test {

    template <typename Type>
    class Scalar {
    public:
        void set(Type value) {
            mScalar.dtype = gdf::test::GdfDataType<Type>::Value;
            gdf::test::setScalar(mScalar, (Type)value);
        }

        gdf_scalar* scalar() {
            return &mScalar;
        }

    public:
        operator int8_t() const {
            return mScalar.data.si08;
        }

        operator int16_t() const {
            return mScalar.data.si16;
        }

        operator int32_t() const {
            return mScalar.data.si32;
        }

        operator int64_t() const {
            return mScalar.data.si64;
        }

        operator uint8_t() const {
            return mScalar.data.ui08;
        }

        operator uint16_t() const {
            return mScalar.data.ui16;
        }

        operator uint32_t() const {
            return mScalar.data.ui32;
        }

        operator uint64_t() const {
            return mScalar.data.ui64;
        }

        operator float() const {
            return mScalar.data.fp32;
        }

        operator double() const {
            return mScalar.data.fp64;
        }

    private:
        gdf_scalar mScalar;
    };

}
}

#endif
