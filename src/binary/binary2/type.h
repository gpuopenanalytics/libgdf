#ifndef GDF_BINARY_TYPE_H
#define GDF_BINARY_TYPE_H

#include "gdf/gdf.h"

namespace gdf {
namespace binary {

    enum class BaseType {
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

    BaseType convertToBaseType(gdf_dtype type);

    const char* getStringFromBaseType(BaseType type);

}
}

#endif
