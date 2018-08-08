#ifndef GDF_BINARY_LIBRARY_OPERATORS_RELATIONAL_CUH
#define GDF_BINARY_LIBRARY_OPERATORS_RELATIONAL_CUH

namespace gdf {
namespace binary {
namespace library {
namespace operators {
namespace relational {

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Equal
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x == y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct NotEqual
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x != y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Less
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x < y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Greater
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x > y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct LessEqual
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x <= y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct GreaterEqual
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x >= y);
        }
    };

}
}
}
}
}

#endif
