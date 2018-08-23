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

}
}
}

#endif
