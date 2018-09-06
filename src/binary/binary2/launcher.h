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
