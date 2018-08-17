#include "binary2/object.h"

namespace gdf {
namespace binary {

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
