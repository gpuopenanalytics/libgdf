#include <jitify.hpp>
#include "gdf/gdf.h"
#include "binary/binary2/type.h"
#include "binary/binary2/kernel.h"
#include "binary/binary2/operation.h"
#include "binary/binary2/traits.h"

namespace gdf {
    namespace {
        static jitify::JitCache kernel_cache;

        dim3 grid(1, 1, 1);
        dim3 block(32, 1, 1);

        std::vector<std::string> compilerFlags { "-std=c++14" };
        std::vector<std::string> headersName  { "operation.h" , "traits.h" };
    }

    std::istream* headersCode(std::string filename, std::iostream& stream) {
        if (filename == "operation.h") {
            stream << gdf::cuda::operation;
            return &stream;
        }
        if (filename == "traits.h") {
            stream << gdf::cuda::traits;
            return &stream;
        }
        return nullptr;
    }

    /*
    auto Instantiate(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
        return
    }
    */
    gdf_error binary_operation(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_binary_operator ope) {
        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
        auto program = kernel_cache.program(gdf::cuda::kernel, headersName, compilerFlags, headersCode);

        auto helper = program.kernel("kernel_v_v_s")
                             .instantiate({"uint32_t" , "uint32_t" , "uint32_t", "Add"})
                             .configure(grid, block);

        auto type = gdf::binary::convertToBaseType(vay->type);
        switch (type) {
            case gdf::binary::BaseType::UI08:
                helper.launch(out->data, vax->data, vay->data.ui08, 32);
                break;
            case gdf::binary::BaseType::UI16:
                helper.launch(out->data, vax->data, vay->data.ui16, 32);
                break;
            case gdf::binary::BaseType::UI32:
                helper.launch(out->data, vax->data, vay->data.ui32, 32);
                break;
            case gdf::binary::BaseType::UI64:
                helper.launch(out->data, vax->data, vay->data.ui64, 32);
                break;
            case gdf::binary::BaseType::SI08:
                helper.launch(out->data, vax->data, vay->data.si08, 32);
                break;
            case gdf::binary::BaseType::SI16:
                helper.launch(out->data, vax->data, vay->data.si16, 32);
                break;
            case gdf::binary::BaseType::SI32:
                helper.launch(out->data, vax->data, vay->data.si32, 32);
                break;
            case gdf::binary::BaseType::SI64:
                helper.launch(out->data, vax->data, vay->data.si64, 32);
                break;
            case gdf::binary::BaseType::FP32:
                helper.launch(out->data, vax->data, vay->data.fp32, 32);
                break;
            case gdf::binary::BaseType::FP64:
                helper.launch(out->data, vax->data, vay->data.fp64, 32);
                break;
        }

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        return GDF_SUCCESS;
    }
}

gdf_error gdf_binary_operation_v_s_v(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, ope);;
}

gdf_error gdf_binary_operation_v_v_s(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, ope);
}

gdf_error gdf_binary_operation_v_v_v(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, ope);
}

gdf_error gdf_binary_operation_v_s_v_d(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, def, ope);
}

gdf_error gdf_binary_operation_v_v_s_d(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, def, ope);
}

gdf_error gdf_binary_operation_v_v_v_d(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, def, ope);
}
