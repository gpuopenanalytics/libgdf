#include "binary/binary2/type.h"

namespace gdf {
namespace binary {

    BaseType convertToBaseType(gdf_dtype type) {
        switch (type) {
            case GDF_INT8:
                return BaseType::SI08;
            case GDF_INT16:
                return BaseType::SI16;
            case GDF_INT32:
            case GDF_DATE32:
                return BaseType::SI32;
            case GDF_INT64:
            case GDF_DATE64:
            case GDF_TIMESTAMP:
                return BaseType::SI64;
            case GDF_FLOAT32:
                return BaseType::FP32;
            case GDF_FLOAT64:
                return BaseType::FP64;
            default:
                return BaseType::FP64;
        }
    }

    const char* getStringFromBaseType(BaseType type) {
        switch (type) {
            case BaseType::UI08:
                return "uint8_t";
            case BaseType::UI16:
                return "uint16_t";
            case BaseType::UI32:
                return "uint32_t";
            case BaseType::UI64:
                return "uint64_t";
            case BaseType::SI08:
                return "int8_t";
            case BaseType::SI16:
                return "int16_t";
            case BaseType::SI32:
                return "int32_t";
            case BaseType::SI64:
                return "int64_t";
            case BaseType::FP32:
                return "float";
            case BaseType::FP64:
                return "double";
        }
    }

}
}
