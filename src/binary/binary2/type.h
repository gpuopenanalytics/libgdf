#ifndef GDF_BINARY_TYPE_H
#define GDF_BINARY_TYPE_H

#include "gdf/gdf.h"

namespace gdf {

    const char* getTypeName(gdf_dtype type);

    const char* getOperatorName(gdf_binary_operator ope);

}

#endif
