#ifndef GDF_GDF_H
#define GDF_GDF_H

#include <cstdlib>
#include <cstdint>
#include "cffi/types.h"
#include "cffi/csv.h"

#define GDF_VALID_BITSIZE (sizeof(gdf_valid_type) * 8)

extern "C" {
#include "cffi/functions.h"
#include "cffi/io_functions.h"
}

#endif /* GDF_GDF_H */
