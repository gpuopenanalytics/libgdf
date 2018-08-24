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
        arguments.push_back(gdf::getTypeName(out->dtype));
        arguments.push_back(gdf::getTypeName(vax->dtype));
        arguments.push_back(gdf::getTypeName(vay->dtype));
        arguments.push_back(gdf::getOperatorName(ope));
        return *this;
    }

    Launcher& Launcher::instantiate(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
        arguments.clear();
        arguments.push_back(gdf::getTypeName(out->dtype));
        arguments.push_back(gdf::getTypeName(vax->dtype));
        arguments.push_back(gdf::getTypeName(vay->dtype));
        arguments.push_back(gdf::getOperatorName(ope));
        return *this;
    }

    Launcher& Launcher::instantiate(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
        arguments.clear();
        arguments.push_back(gdf::getTypeName(out->dtype));
        arguments.push_back(gdf::getTypeName(vax->dtype));
        arguments.push_back(gdf::getTypeName(vay->dtype));
        arguments.push_back(gdf::getTypeName(def->dtype));
        arguments.push_back(gdf::getOperatorName(ope));
        return *this;
    }

    Launcher& Launcher::instantiate(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        arguments.clear();
        arguments.push_back(gdf::getTypeName(out->dtype));
        arguments.push_back(gdf::getTypeName(vax->dtype));
        arguments.push_back(gdf::getTypeName(vay->dtype));
        arguments.push_back(gdf::getTypeName(def->dtype));
        arguments.push_back(gdf::getOperatorName(ope));
        return *this;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_scalar* vay) {
        program.kernel(kernelName.c_str())
               .instantiate(arguments)
               .configure_1d_max_occupancy()
               .launch(out->size, out->data, vax->data, vay->data);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_column* vay) {
        program.kernel(kernelName.c_str())
               .instantiate(arguments)
               .configure_1d_max_occupancy()
               .launch(out->size, out->data, vax->data, vay->data);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def)
    {
        program.kernel(kernelName)
               .instantiate(arguments)
               .configure_1d_max_occupancy()
               .launch(out->size, def->data,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def)
    {
        program.kernel(kernelName)
               .instantiate(arguments)
               .configure_1d_max_occupancy()
               .launch(out->size, def->data,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid, vay->valid);

        return GDF_SUCCESS;
    }
}
