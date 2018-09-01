#ifndef GDF_LIBRARY_VECTOR_H
#define GDF_LIBRARY_VECTOR_H

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "gdf/gdf.h"
#include "library/types.h"

namespace gdf {
namespace library {

    template <typename Type>
    class Vector {
    public:
        using ValidType = int32_t;
        static constexpr int ValidSize = 32;

    public:
        ~Vector() {
            eraseGpu();
        }

        void eraseGpu() {
            if (isDataAllocated) {
                isDataAllocated = false;
                cudaFree(mColumn.data);
            }
            if (isValidAllocated) {
                isValidAllocated = false;
                cudaFree(mColumn.valid);
            }
        }

        Vector& clear() {
            eraseGpu();
            mData.clear();
            mValid.clear();
            return *this;
        }

        Vector& range(Type init, Type final, Type step) {
            assert(0 < step);
            assert(init < final);

            int size = ((final - init) / step) + 1;
            mData.clear();
            mData.reserve(size);

            while (init < final) {
                mData.push_back(init);
                init += step;
            }

            copyDataToGpu();
            return *this;
        }

        Vector& fill(int size, Type value) {
            mData.clear();
            mData.reserve(size);

            for (int k = 0; k < size; ++k) {
                mData.push_back(value);
            }

            copyDataToGpu();
            return *this;
        }

        Vector& valid(bool value) {
            assert(mData.size() != 0);
            mValid.clear();
            int size = (mData.size() / ValidSize) + ((mData.size() % ValidSize) ? 1 : 0);
            std::generate_n(std::back_inserter(mValid), size, [value] { return -(ValidType)value; });
            return *this;
        }

        // TODO: implementation incomplete
        Vector& valid(bool value, std::initializer_list<int> list) {
            assert(mData.size() != 0);
            assert(mValid.size() != 0);
            mValid.clear();
            return *this;
        }

        Vector& valid(bool value, int init, int final, int step) {
            mValid.clear();
            int size = (mData.size() / ValidSize) + ((mData.size() % ValidSize) ? 1 : 0);
            for (int index = 0; index < size; ++index) {
                ValidType val = 0;
                while (((init / ValidSize) == index) && (init < final)) {
                    val |= (1 << (init % ValidSize));
                    init += step;
                }
                if (value) {
                    mValid.push_back(val);
                } else {
                    mValid.push_back(~val);
                }
            }
            copyValidToGpu();
            return *this;
        }

        void emplace(int size) {
            int dataSize = size;
            int validSize = (dataSize / ValidSize) + ((dataSize % ValidSize) ? 1 : 0);

            std::generate_n(std::back_inserter(mData), dataSize, [] { return 0; });
            std::generate_n(std::back_inserter(mValid), validSize, [] { return 0; });

            allocDataInGpu();
            allocValidInGpu();
        }

        void read() {
            if (isDataAllocated) {
                cudaMemcpy(mData.data(), mColumn.data, mData.size() * sizeof(Type), cudaMemcpyDeviceToHost);
            }
            if (isValidAllocated) {
                cudaMemcpy(mValid.data(), mColumn.valid, mValid.size() * sizeof(ValidType), cudaMemcpyDeviceToHost);
            }
        }

    public:
        int dataSize() {
            return mData.size();
        }

        int validSize() {
            return mValid.size();
        }

        gdf_column* column() {
            return &mColumn;
        }

        Type data(int index) {
            return mData[index];
        }

        ValidType valid(int index) {
            return mValid[index];
        }

    private:
        void allocDataInGpu() {
            isDataAllocated = true;
            mColumn.size = mData.size();
            mColumn.dtype = GdfDataType<Type>::Value;
            cudaMalloc((void**)&(mColumn.data), mData.size() * sizeof(Type));
        }

        void allocValidInGpu() {
            isValidAllocated = true;
            cudaMalloc((void**)&(mColumn.valid), mValid.size() * sizeof(ValidType));
        }

        void copyDataToGpu() {
            isDataAllocated = true;
            mColumn.size = mData.size();
            mColumn.dtype = GdfDataType<Type>::Value;
            cudaMalloc((void**)&(mColumn.data), mData.size() * sizeof(Type));
            cudaMemcpy(mColumn.data, mData.data(), mData.size() * sizeof(Type), cudaMemcpyHostToDevice);
        }

        void copyValidToGpu() {
            isValidAllocated = true;
            cudaMalloc((void**)&(mColumn.valid), mValid.size() * sizeof(ValidType));
            cudaMemcpy(mColumn.valid, mValid.data(), mValid.size() * sizeof(ValidType), cudaMemcpyHostToDevice);
        }

    private:
        std::vector<Type> mData;
        std::vector<ValidType> mValid;

    private:
        gdf_column mColumn;

    private:
        bool isDataAllocated {false};
        bool isValidAllocated {false};
    };
}
}

#endif
