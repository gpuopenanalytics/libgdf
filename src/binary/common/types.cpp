#include "binary/common/types.h"

namespace gdf {
namespace binary {
namespace common {

    GdfBaseType getGdfBaseType(gdf_dtype type) {
        switch (type) {
            case GDF_INT8:
                return GdfBaseType::SI08;
            case GDF_INT16:
                return GdfBaseType::SI16;
            case GDF_INT32:
            case GDF_DATE32:
                return GdfBaseType::SI32;
            case GDF_INT64:
            case GDF_DATE64:
            case GDF_TIMESTAMP:
                return GdfBaseType::SI64;
            case GDF_FLOAT32:
                return GdfBaseType::FP32;
            case GDF_FLOAT64:
                return GdfBaseType::FP64;
            default:
                return GdfBaseType::FP64;
        }
    }

    GdfObject::GdfObject(gdf_scalar* value)
    : data {value},
      type {GdfType::Scalar},
      base {getGdfBaseType(value->type)}
    { }

    GdfObject::GdfObject(gdf_column* value)
    : data {value},
      type {GdfType::Vector},
      base {getGdfBaseType(value->dtype)}
    { }

    gdf_scalar* GdfObject::getScalar() {
        return data.scalar;
    }

    gdf_column* GdfObject::getVector() {
        return data.vector;
    }

    GdfType GdfObject::getType() {
        return type;
    }

    GdfBaseType GdfObject::getBaseType() {
        return base;
    }

    GdfObject::GdfData::GdfData(gdf_scalar* value)
    : scalar {value}
    { }

    GdfObject::GdfData::GdfData(gdf_column* value)
    : vector {value}
    { }

}
}
}
