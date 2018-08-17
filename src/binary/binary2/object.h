#ifndef GDF_BINARY_OBJECT_H
#define GDF_BINARY_OBJECT_H

#include "binary2/type.h"

namespace gdf {
namespace binary {

    class GdfObject {
    public:
        enum class GdfType {
            Scalar,
            Vector
        };

    public:
        GdfObject(gdf_scalar* value);

        GdfObject(gdf_column* value);

    public:
        gdf_scalar* getScalar();

        gdf_column* getVector();

    public:
        GdfType getGdfType();

        BaseType getBaseType();

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
        BaseType base;
    };

}
}

#endif
