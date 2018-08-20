#include "gdf/gdf.h"
#include "binary/binary2/launcher.h"

namespace gdf {
    gdf_error operation_launch(Launcher& launcher, gdf_column* out, gdf_column* vax, gdf_scalar* vay) {
        auto type = gdf::convertToBaseType(vay->type);
        switch (type) {
            case gdf::BaseType::UI08:
                launcher.launch(out, vax, vay->data.ui08);
                break;
            case gdf::BaseType::UI16:
                launcher.launch(out, vax, vay->data.ui16);
                break;
            case gdf::BaseType::UI32:
                launcher.launch(out, vax, vay->data.ui32);
                break;
            case gdf::BaseType::UI64:
                launcher.launch(out, vax, vay->data.ui64);
                break;
            case gdf::BaseType::SI08:
                launcher.launch(out, vax, vay->data.si08);
                break;
            case gdf::BaseType::SI16:
                launcher.launch(out, vax, vay->data.si16);
                break;
            case gdf::BaseType::SI32:
                launcher.launch(out, vax, vay->data.si32);
                break;
            case gdf::BaseType::SI64:
                launcher.launch(out, vax, vay->data.si64);
                break;
            case gdf::BaseType::FP32:
                launcher.launch(out, vax, vay->data.fp32);
                break;
            case gdf::BaseType::FP64:
                launcher.launch(out, vax, vay->data.fp64);
                break;
        }

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
        gdf::Launcher launcher;

        launcher.kernel("kernel_v_v_s")
                .instantiate(out, vax, vay, ope)
                .configure(dim3(1, 1, 1), dim3(32, 1, 1));

        return operation_launch(launcher, out, vax, vay);
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
        gdf::Launcher launcher;

        launcher.kernel("kernel_v_v_v")
                .instantiate(out, vax, vay, ope)
                .configure(dim3(1, 1, 1), dim3(32, 1, 1))
                .launch(out, vax, vay);

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
