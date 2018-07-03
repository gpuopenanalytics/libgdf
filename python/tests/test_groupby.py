from __future__ import print_function
import ctypes
from contextlib import contextmanager, ExitStack

import pytest

import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf

from .utils import new_column, unwrap_devary, get_dtype


@contextmanager
def make_column(cpu_data):
    device_data = cuda.to_device(cpu_data)
    column = new_column()
    libgdf.gdf_column_view(column, unwrap_devary(device_data), ffi.NULL,
                           device_data.size, get_dtype(device_data.dtype))
    yield column



params_dtypes = [np.int8, np.int32, np.int64, np.float32, np.float64]


@pytest.mark.parametrize('dtype1', params_dtypes)
@pytest.mark.parametrize('dtype2', params_dtypes)
@pytest.mark.parametrize('nelem', [1, 2, 10, 100, 1001, 10003])
def test_order_by(dtype1, dtype2, nelem):
    # Make data
    with ExitStack() as stack:
        data = [
            np.random.randint(0, 3, nelem).astype(dtype1),
            np.random.randint(0, 10, nelem).astype(dtype2),
        ]
        cols = [stack.enter_context(make_column(d)) for d in data]

        columns = ffi.new('gdf_column[]', len(cols))
        for i, col in enumerate(cols):
            columns[i] = col[0]

        d_cols = cuda.device_array(len(cols) * ffi.sizeof('void*'),
                                   dtype=np.byte)
        d_types = cuda.device_array(len(cols) * ffi.sizeof('gdf_dtype'),
                                    dtype=np.byte)
        d_indx = cuda.device_array(nelem, dtype=np.intp)

        # Call libgdf
        libgdf.gdf_order_by(
            nelem,
            columns,
            len(cols),
            unwrap_devary(d_cols),
            unwrap_devary(d_types),
            unwrap_devary(d_indx),
        )

        expect = np.lexsort((data[1], data[0]))
        print(expect)
        got = d_indx.copy_to_host()
        print(got)
        # Check result
        np.testing.assert_equal(got, expect)
