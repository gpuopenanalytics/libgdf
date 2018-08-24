#ifndef GDF_BINARY_LAUNCHER_H
#define GDF_BINARY_LAUNCHER_H

#include <jitify.hpp>
#include "binary/binary2/type.h"

namespace gdf {

    std::istream* headersCode(std::string filename, std::iostream& stream);

    class Launcher {
    public:
        static Launcher launch() {
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

        Launcher& instantiate(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope);

        Launcher& instantiate(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope);

        gdf_error launch(gdf_column* out, gdf_column* vax, gdf_scalar* vay);

        gdf_error launch(gdf_column* out, gdf_column* vax, gdf_column* vay);

        gdf_error launch(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def);

        gdf_error launch(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def);

    private:
        std::vector<std::string> compilerFlags { "-std=c++14" };
        std::vector<std::string> headersName { "operation.h" , "traits.h" , "kernel_gdf_data.h" };

    private:
        jitify::Program program;

    private:
        std::string kernelName;
        std::vector<std::string> arguments;
    };
}

#endif
