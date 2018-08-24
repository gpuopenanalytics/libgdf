#ifndef GDF_TEST_LIBRARY_OPERATION_H
#define GDF_TEST_LIBRARY_OPERATION_H

namespace gdf {
namespace test {
namespace operation {

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Add {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax + (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Sub {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax - (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mul {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax * (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Div {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax / (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct TrueDiv {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            return (TypeOut)((double)vax / (double)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct FloorDiv {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            return (TypeOut)floor((double)vax / (double)vay);
        }
    };

    template <typename TypeOut,
              typename TypeVax,
              typename TypeVay,
              typename Common = typename std::common_type<TypeVax, TypeVay>::type>
    struct Mod;

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mod<TypeOut, TypeVax, TypeVay, uint64_t> {
        TypeOut operator()(TypeVax x, TypeVay y) {
            return (TypeOut)((uint64_t)x % (uint64_t)y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mod<TypeOut, TypeVax, TypeVay, float> {
        TypeOut operator()(TypeVax x, TypeVay y) {
            return (TypeOut)fmod((float)x, (float)y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mod<TypeOut, TypeVax, TypeVay, double> {
        TypeOut operator()(TypeVax x, TypeVay y) {
            return (TypeOut)fmod((double)x, (double)y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Pow {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            return (TypeOut)pow((double)vax, (double)vay);
        }
    };
}
}
}

#endif
