#include "gdf/gdf.h"
#include "binary/binary2/launcher.h"

namespace gdf {
    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_s")
                               .instantiate(out, vax, vay, ope)
                               .launch(out, vax, vay);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_v")
                               .instantiate(out, vax, vay, ope)
                               .launch(out, vax, vay);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_s_d")
                               .instantiate(out, vax, vay, def, ope)
                               .launch(out, vax, vay, def);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        gdf::Launcher::launch().kernel("kernel_v_v_d")
                               .instantiate(out, vax, vay, def, ope)
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
