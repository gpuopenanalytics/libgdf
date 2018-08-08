#include "gdf/gdf.h"
#include "binary/common/types.h"
#include "binary/common/selector.h"

namespace gdf {
namespace binary {
namespace common {

    static gdf_error inputError = GDF_SUCCESS;

    gdf_error getInputError() {
        return inputError;
    }

    bool verifyInputScalar(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator operation) {
        if (out->size != vax->size) {
            inputError = GDF_COLUMN_SIZE_MISMATCH;
            return true;
        }
        return false;
    }

    bool verifyInputVector(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator operation) {
        if (vax->size != vay->size) {
            inputError = GDF_COLUMN_SIZE_MISMATCH;
            return true;
        }
        if (vax->size != out->size) {
            inputError = GDF_COLUMN_SIZE_MISMATCH;
            return true;
        }
        return false;
    }

    gdf_error executeBinaryOperation(gdf_column* out, gdf_column* vax, GdfObject* vay, gdf_scalar* def, gdf_binary_operator ope) {
        return select(out, vax, vay, def, ope);
    }
}
}
}


gdf_error gdf_scalar_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
    namespace space = gdf::binary::common;
    if (space::verifyInputScalar(out, vax, vay, ope)) {
        return space::getInputError();
    }

    space::GdfObject obj(vay);
    return space::executeBinaryOperation(out, vax, &obj, def, ope);
}

gdf_error gdf_vector_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
    namespace space = gdf::binary::common;
    if (space::verifyInputVector(out, vax, vay, ope)) {
        return space::getInputError();
    }

    space::GdfObject obj(vay);
    return space::executeBinaryOperation(out, vax, &obj, def, ope);
}
