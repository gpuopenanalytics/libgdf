#ifndef GDF_BINARY_LAUNCHER_H
#define GDF_BINARY_LAUNCHER_H

#include <jitify.hpp>
#include "binary/binary2/type.h"

namespace gdf {

    std::istream* headersCode(std::string filename, std::iostream& stream);

    class Launcher {
    public:
        static Launcher Launch() {
            return Launcher();
        }

    public:
        Launcher();

        Launcher(Launcher&&);

    public:
        Launcher(const Launcher&) = delete;

        Launcher& operator=(Launcher&&) = delete;

        Launcher& operator=(const Launcher&) = delete;

    public:
        Launcher& kernel(std::string&& value);

        Launcher& instantiate(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope);

        Launcher& instantiate(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope);

        Launcher& configure(dim3 grid, dim3 block);

        template <typename Type>
        gdf_error launch(gdf_column* out, gdf_column* vax, Type vay) {
            program.kernel(kernelName)
                   .instantiate(arguments)
                   .configure(grid, block)
                   .launch(out->data, vax->data, vay, out->size);

            return GDF_SUCCESS;
        }

        gdf_error launch(gdf_column* out, gdf_column* vax, gdf_column* vay) {
            program.kernel(kernelName.c_str())
                   .instantiate(arguments)
                   .configure(grid, block)
                   .launch(out->data, vax->data, vay->data, out->size);

            return GDF_SUCCESS;
        }

    private:
        std::vector<std::string> compilerFlags { "-std=c++14" };
        std::vector<std::string> headersName { "operation.h" , "traits.h" };

    private:
        jitify::Program program;

    private:
        dim3 grid;
        dim3 block;
        std::string kernelName;
        std::vector<std::string> arguments;
    };
}

#endif
