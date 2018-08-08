#ifndef GDF_BINARY_COMMON_TYPES_H
#define GDF_BINARY_COMMON_TYPES_H

#include "gdf/gdf.h"

namespace gdf {
namespace binary {
namespace common {

    enum class GdfBaseType {
        UI08,
        UI16,
        UI32,
        UI64,
        SI08,
        SI16,
        SI32,
        SI64,
        FP32,
        FP64
    };

    GdfBaseType getGdfBaseType(gdf_dtype type);

    enum class GdfType {
        Scalar,
        Vector
        //Matrix
    };

    class GdfObject {
    public:
        GdfObject(gdf_scalar* value);

        GdfObject(gdf_column* value);

    public:
        gdf_scalar* getScalar();

        gdf_column* getVector();

    public:
        GdfType getType();

        GdfBaseType getBaseType();

    private:
        union GdfData {
            GdfData(gdf_scalar* value);
            GdfData(gdf_column* value);

            gdf_scalar* scalar;
            gdf_column* vector;
        };

    private:
        GdfData data;
        GdfType type;
        GdfBaseType base;
    };

}
}
}

#endif
