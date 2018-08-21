#include "binary/binary2/launcher.h"
#include "binary/binary2/cuda.h"

namespace gdf {
    static jitify::JitCache JitCache;

    std::istream* headersCode(std::string filename, std::iostream& stream) {
        if (filename == "operation.h") {
            stream << gdf::cuda::operation;
            return &stream;
        }
        if (filename == "traits.h") {
            stream << gdf::cuda::traits;
            return &stream;
        }
        if (filename == "kernel_gdf_data.h") {
            stream << gdf::cuda::kernel_gdf_data;
            return &stream;
        }
        return nullptr;
    }

    Launcher::Launcher()
     : program {JitCache.program(gdf::cuda::kernel, headersName, compilerFlags, gdf::headersCode)}
    { }

    Launcher::Launcher(Launcher&& launcher)
     : program {std::move(launcher.program)}
    { }

    Launcher& Launcher::kernel(std::string&& value) {
        kernelName = value;
        return *this;
    }

    Launcher& Launcher::instantiate(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
        arguments.clear();
        arguments.push_back(gdf::getStringFromBaseType(gdf::convertToBaseType(out->dtype)));
        arguments.push_back(gdf::getStringFromBaseType(gdf::convertToBaseType(vax->dtype)));
        arguments.push_back(gdf::getStringFromBaseType(gdf::convertToBaseType(vay->type)));
        arguments.push_back(gdf::getOperatorName(ope));
        return *this;
    }

    Launcher& Launcher::instantiate(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
        arguments.clear();
        arguments.push_back(getStringFromBaseType(convertToBaseType(out->dtype)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(vax->dtype)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(vay->dtype)));
        arguments.push_back(getOperatorName(ope));
        return *this;
    }

    Launcher& Launcher::instantiate(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
        arguments.clear();
        arguments.push_back(getStringFromBaseType(convertToBaseType(out->dtype)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(vax->dtype)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(vay->type)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(def->type)));
        arguments.push_back(getOperatorName(ope));
        return *this;
    }

    Launcher& Launcher::instantiate(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        arguments.clear();
        arguments.push_back(getStringFromBaseType(convertToBaseType(out->dtype)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(vax->dtype)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(vay->dtype)));
        arguments.push_back(getStringFromBaseType(convertToBaseType(def->type)));
        arguments.push_back(getOperatorName(ope));
        return *this;
    }

    Launcher& Launcher::configure(dim3 grid, dim3 block) {
        this->grid = grid;
        this->block = block;
        return *this;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_scalar* vay) {
        program.kernel(kernelName.c_str())
               .instantiate(arguments)
               .configure(grid, block)
               .launch(out->size,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_column* vay) {
        program.kernel(kernelName.c_str())
               .instantiate(arguments)
               .configure(grid, block)
               .launch(out->size,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid, vay->valid);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def)
    {
        program.kernel(kernelName)
               .instantiate(arguments)
               .configure(grid, block)
               .launch(out->size, def->data,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def)
    {
        program.kernel(kernelName)
               .instantiate(arguments)
               .configure(grid, block)
               .launch(out->size, def->data,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid, vay->valid);

        return GDF_SUCCESS;
    }
}
