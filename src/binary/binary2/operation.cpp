namespace gdf {
namespace cuda {

const char* operation =
R"***(
#pragma once

    template <typename ConcreteOperation>
    struct AbstractOperation {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return static_cast<ConcreteOperation*>(this)->template operate<TypeOut, TypeVax, TypeVay>(x, y);
        }
    };

    struct Add : public AbstractOperation<Add> {
        template <typename TypeOut,
                  typename TypeVax,
                  typename TypeVay,
                  typename Common = CommonNumber<TypeVax, TypeVay>,
                  enableIf<(isIntegralSigned<Common>)>* = nullptr>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)((Common)x + (Common)y);
        }

        template <typename TypeOut,
                  typename TypeVax,
                  typename TypeVay,
                  typename Common = CommonNumber<TypeVax, TypeVay>,
                  enableIf<(isIntegralUnsigned<Common>)>* = nullptr>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)((Common)x + (Common)y);
        }

        template <typename TypeOut,
                  typename TypeVax,
                  typename TypeVay,
                  typename Common = CommonNumber<TypeVax, TypeVay>,
                  enableIf<(isFloatingPoint<Common>)>* = nullptr>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)((Common)x + (Common)y);
        }
    };
)***";
}
}
