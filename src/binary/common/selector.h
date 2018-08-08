#ifndef GDF_BINARY_COMMON_SELECTOR_H
#define GDF_BINARY_COMMON_SELECTOR_H

#include "gdf/gdf.h"
#include "binary/common/operations.cuh"
#include "binary/library/operators/arithmetic.cuh"
#include "binary/library/operators/relational.cuh"

namespace gdf {
namespace binary {
namespace common {

    namespace {
        template <typename TypeOut, typename TypeVax, typename TypeVay>
        gdf_error selectTypeOpe(gdf_column* out, gdf_column* vax, GdfObject* vay, gdf_scalar* def, gdf_binary_operator ope) {
            namespace arithmetic_operators = gdf::binary::library::operators::arithmetic;
            namespace relational_operators = gdf::binary::library::operators::relational;

            switch (ope) {
                case GDF_ADD:
                    using Add = arithmetic_operators::Add<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Add>(out, vax, vay, def, Add());
                case GDF_SUB:
                    using Sub = arithmetic_operators::Sub<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Sub>(out, vax, vay, def, Sub());
                case GDF_MUL:
                    using Mul = arithmetic_operators::Mul<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Mul>(out, vax, vay, def, Mul());
                case GDF_DIV:
                    using Div = arithmetic_operators::Div<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Div>(out, vax, vay, def, Div());
                case GDF_TRUE_DIV:
                    using TrueDiv = arithmetic_operators::TrueDiv<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, TrueDiv>(out, vax, vay, def, TrueDiv());
                case GDF_FLOOR_DIV:
                    using FloorDiv = arithmetic_operators::FloorDiv<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, FloorDiv>(out, vax, vay, def, FloorDiv());
                case GDF_MOD:
                    using Mod = arithmetic_operators::Mod<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Mod>(out, vax, vay, def, Mod());
                case GDF_POW:
                    using Pow = arithmetic_operators::Pow<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Pow>(out, vax, vay, def, Pow());
                case GDF_EQUAL:
                    using Equal = relational_operators::Equal<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Equal>(out, vax, vay, def, Equal());
                case GDF_NOT_EQUAL:
                    using NotEqual = relational_operators::NotEqual<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, NotEqual>(out, vax, vay, def, NotEqual());
                case GDF_LESS:
                    using Less = relational_operators::Less<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Less>(out, vax, vay, def, Less());
                case GDF_GREATER:
                    using Greater = relational_operators::Greater<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, Greater>(out, vax, vay, def, Greater());
                case GDF_LESS_EQUAL:
                    using LessEqual = relational_operators::LessEqual<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, LessEqual>(out, vax, vay, def, LessEqual());
                case GDF_GREATER_EQUAL:
                    using GreaterEqual = relational_operators::GreaterEqual<TypeOut, TypeVax, TypeVay>;
                    return operate<TypeOut, TypeVax, TypeVay, GreaterEqual>(out, vax, vay, def, GreaterEqual());
                default:
                    return GDF_UNSUPPORTED_DTYPE;
            }
        }


        template <typename TypeOut, typename TypeVax>
        gdf_error selectTypeVay(gdf_column* out, gdf_column* vax, GdfObject* vay, gdf_scalar* def, gdf_binary_operator ope) {
            switch (vay->getBaseType()) {
                case GdfBaseType::SI08:
                    return selectTypeOpe<TypeOut, TypeVax, int8_t>(out, vax, vay, def, ope);
                case GdfBaseType::SI16:
                    return selectTypeOpe<TypeOut, TypeVax, int16_t>(out, vax, vay, def, ope);
                case GdfBaseType::SI32:
                    return selectTypeOpe<TypeOut, TypeVax, int32_t>(out, vax, vay, def, ope);
                case GdfBaseType::SI64:
                    return selectTypeOpe<TypeOut, TypeVax, int64_t>(out, vax, vay, def, ope);
                case GdfBaseType::FP32:
                    return selectTypeOpe<TypeOut, TypeVax, float>(out, vax, vay, def, ope);
                case GdfBaseType::FP64:
                    return selectTypeOpe<TypeOut, TypeVax, double>(out, vax, vay, def, ope);
                default:
                    return GDF_UNSUPPORTED_DTYPE;
            }
        }


        template <typename TypeOut>
        gdf_error selectTypeVax(gdf_column* out, gdf_column* vax, GdfObject* vay, gdf_scalar* def, gdf_binary_operator ope) {
            switch (vax->dtype) {
                case GDF_INT8:
                    return selectTypeVay<TypeOut, int8_t>(out, vax, vay, def, ope);
                case GDF_INT16:
                    return selectTypeVay<TypeOut, int16_t>(out, vax, vay, def, ope);
                case GDF_INT32:
                case GDF_DATE32:
                    return selectTypeVay<TypeOut, int32_t>(out, vax, vay, def, ope);
                case GDF_INT64:
                case GDF_DATE64:
                case GDF_TIMESTAMP:
                    return selectTypeVay<TypeOut, int64_t>(out, vax, vay, def, ope);
                case GDF_FLOAT32:
                    return selectTypeVay<TypeOut, float>(out, vax, vay, def, ope);
                case GDF_FLOAT64:
                    return selectTypeVay<TypeOut, double>(out, vax, vay, def, ope);
                default:
                    return GDF_UNSUPPORTED_DTYPE;
            }
        }
    }


    gdf_error select(gdf_column* out, gdf_column* vax, GdfObject* vay, gdf_scalar* def, gdf_binary_operator ope) {
        switch (out->dtype) {
            case GDF_INT8:
                return selectTypeVax<int8_t>(out, vax, vay, def, ope);
            case GDF_INT16:
                return selectTypeVax<int16_t>(out, vax, vay, def, ope);
            case GDF_INT32:
            case GDF_DATE32:
                return selectTypeVax<int32_t>(out, vax, vay, def, ope);
            case GDF_INT64:
            case GDF_DATE64:
            case GDF_TIMESTAMP:
                return selectTypeVax<int64_t>(out, vax, vay, def, ope);
            case GDF_FLOAT32:
                return selectTypeVax<float>(out, vax, vay, def, ope);
            case GDF_FLOAT64:
                return selectTypeVax<double>(out, vax, vay, def, ope);
            default:
                return GDF_UNSUPPORTED_DTYPE;
        }
    }

}
}
}

#endif
