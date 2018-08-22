namespace gdf {
namespace cuda {

const char* kernel_gdf_data =
R"***(
#pragma once

    union gdf_data {
        int8_t   si08;
        int16_t  si16;
        int32_t  si32;
        int64_t  si64;
        uint8_t  ui08;
        uint16_t ui16;
        uint32_t ui32;
        uint64_t ui64;
        float    fp32;
        double   fp64;


        operator int8_t() const {
            return si08;
        }

        operator int16_t() const {
            return si16;
        }

        operator int32_t() const {
            return si32;
        }

        operator int64_t() const {
            return si64;
        }

        operator uint8_t() const {
            return ui08;
        }

        operator uint16_t() const {
            return ui16;
        }

        operator uint32_t() const {
            return ui32;
        }

        operator uint64_t() const {
            return ui64;
        }

        operator float() const {
            return fp32;
        }

        operator double() const {
            return fp64;
        }
    };

)***";
}
}
