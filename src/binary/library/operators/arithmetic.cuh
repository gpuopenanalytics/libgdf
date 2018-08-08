#ifndef GDF_LIBRARY_OPERATORS_ARITHMETIC_CUH
#define GDF_LIBRARY_OPERATORS_ARITHMETIC_CUH

namespace gdf {
namespace binary {
namespace library {
namespace operators {
namespace arithmetic {

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Add : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x + y);
        }
    };


    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Sub : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x - y);
        }
    };


    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mul : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x * y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Div : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)(x / y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct TrueDiv : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)((double)x / (double)y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct FloorDiv : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)floor(x / y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mod : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)((int64_t)(x) % y);
        }
    };

    template <typename TypeOut, typename TypeVax>
    struct Mod<TypeOut, TypeVax, float> : public thrust::binary_function<TypeVax, float, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, float y)
        {
            return (TypeOut)fmod((float)x, y);
        }
    };

    template <typename TypeOut, typename TypeVax>
    struct Mod<TypeOut, TypeVax, double> : public thrust::binary_function<TypeVax, double, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, double y)
        {
            return (TypeOut)fmod((double)x, y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Pow : public thrust::binary_function<TypeVax, TypeVay, TypeOut>
    {
        __host__ __device__
        TypeOut operator()(TypeVax x, TypeVay y)
        {
            return (TypeOut)pow((double)x, (double)y);
        }
    };

}
}
}
}
}

#endif
