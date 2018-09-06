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

#include "gdf/gdf.h"
#include "binary/binary2/launcher.h"

namespace gdf {
    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_s")
                               .instantiate(ope, out, vax, vay)
                               .launch(out, vax, vay);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_v")
                               .instantiate(ope, out, vax, vay)
                               .launch(out, vax, vay);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_s_d")
                               .instantiate(ope, out, vax, vay, def)
                               .launch(out, vax, vay, def);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_v_d")
                               .instantiate(ope, out, vax, vay, def)
                               .launch(out, vax, vay, def);

        return GDF_SUCCESS;
    }
}


gdf_error gdf_binary_operation_v_s_v(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vay, vax, ope);
}

gdf_error gdf_binary_operation_v_v_s(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, ope);
}

gdf_error gdf_binary_operation_v_v_v(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, ope);
}

gdf_error gdf_binary_operation_v_s_v_d(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vay, vax, def, ope);
}

gdf_error gdf_binary_operation_v_v_s_d(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, def, ope);
}

gdf_error gdf_binary_operation_v_v_v_d(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binary_operation(out, vax, vay, def, ope);
}
