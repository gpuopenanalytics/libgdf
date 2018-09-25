/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace gdf {
namespace binops {
namespace jit {
namespace code {

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
/*
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
*/
    struct Add : public AbstractOperation<Add> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x + (Common)y);
        }
    };

    struct Sub : public AbstractOperation<Sub> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x - (Common)y);
        }
    };

    struct Mul : public AbstractOperation<Mul> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x * (Common)y);
        }
    };

    struct Div : public AbstractOperation<Div> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x / (Common)y);
        }
    };

    struct TrueDiv : public AbstractOperation<TrueDiv> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)((double)x / (double)y);
        }
    };

    struct FloorDiv : public AbstractOperation<FloorDiv> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)floor((double)x / (double)y);
        }
    };

    struct Mod : public AbstractOperation<Mod> {
        template <typename TypeOut,
                  typename TypeVax,
                  typename TypeVay,
                  typename Common = CommonNumber<TypeVax, TypeVay>,
                  enableIf<(isIntegral<Common>)>* = nullptr>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)((Common)x % (Common)y);
        }

        template <typename TypeOut,
                  typename TypeVax,
                  typename TypeVay,
                  typename Common = CommonNumber<TypeVax, TypeVay>,
                  enableIf<(isFloat<Common>)>* = nullptr>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)fmodf((Common)x, (Common)y);
        }

        template <typename TypeOut,
                  typename TypeVax,
                  typename TypeVay,
                  typename Common = CommonNumber<TypeVax, TypeVay>,
                  enableIf<(isDouble<Common>)>* = nullptr>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)fmod((Common)x, (Common)y);
        }
    };

    struct Pow : public AbstractOperation<Pow> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            return (TypeOut)pow((double)x, (double)y);
        }
    };

    struct Equal : public AbstractOperation<Equal> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x == (Common)y);
        }
    };

    struct NotEqual : public AbstractOperation<NotEqual> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x != (Common)y);
        }
    };

    struct Less : public AbstractOperation<Less> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x < (Common)y);
        }
    };

    struct Greater : public AbstractOperation<Greater> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x > (Common)y);
        }
    };

    struct LessEqual : public AbstractOperation<LessEqual> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x <= (Common)y);
        }
    };

    struct GreaterEqual : public AbstractOperation<GreaterEqual> {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        __device__
        TypeOut operate(TypeVax x, TypeVay y) {
            using Common = CommonNumber<TypeVax, TypeVay>;
            return (TypeOut)((Common)x >= (Common)y);
        }
    };
)***";

} // code
} // jit
} // binops
} // gdf
