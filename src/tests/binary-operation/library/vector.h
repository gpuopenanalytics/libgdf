#ifndef GDF_TEST_LIBRARY_VECTOR_H
#define GDF_TEST_LIBRARY_VECTOR_H

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "gdf/gdf.h"
#include "library/types.h"

namespace gdf {
namespace test {

    template <typename Type>
    class Vector {
    public:
        using ValidType = uint32_t;

        static constexpr int ValidSize = 32;

    private:
        enum class State {
            Init,
            Cpu,
            Gpu,
            GpuAlloc,
            GpuCopy,
        };

    public:
        ~Vector() {
            if (mDataState == State::Gpu) {
                cudaFree(mColumn.data);
            }
            if (mValidState == State::Gpu) {
                cudaFree(mColumn.valid);
            }
        }

        Vector& range(int init, int final, int step) {
            int size = ((final - init) / step) + 1;
            mData.clear();
            mData.reserve(size);
            while (init < final) {
                mData.push_back((Type)init);
                init += step;
            }
            mDataState = State::GpuCopy;
            emplace();
            return *this;
        }

        Vector& fill(int size, int value) {
            mData.clear();
            mData.reserve(size);
            for (int k = 0; k < size; ++k) {
                mData.push_back((Type)value);
            }
            mDataState = State::GpuCopy;
            emplace();
            return *this;
        }

        Vector& valid(bool value, std::initializer_list<int> list) {
            assert(mData.size() != 0);
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
            mValidState = State::GpuCopy;
            emplace();
            return *this;
        }

        void emplace(int size) {
            int dataSize = size;
            int validSize = (dataSize / ValidSize) + ((dataSize % ValidSize) ? 1 : 0);

            std::generate_n(std::back_inserter(mData), dataSize, [] { return 0; });
            std::generate_n(std::back_inserter(mValid), validSize, [] { return 0; });

            mDataState = State::GpuAlloc;
            mValidState = State::GpuAlloc;
            emplace();
        }

        void read() {
            if (mDataState == State::Gpu) {
                cudaMemcpy(mData.data(), mColumn.data, mData.size() * sizeof(Type), cudaMemcpyDeviceToHost);
            }
            if (mValidState == State::Gpu) {
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

    public:
        std::string toString(int size = -1) {
            if (size == -1) {
                size = 16;
            }

            std::stringstream ss;
            ss << "vector: {\n";
            ss << "{size : " << mColumn.size << " | " << mData.size() << "},\n"
               << "{dtype : " << mColumn.dtype << " | " << getTypeName(mColumn.dtype) << " }\n";
            ss << "{data : \n";
            for (int k = 0; k < (uint64_t)mData.size(); ++k) {
                if (!(k % size) && (k != 0)) {
                    ss << "\n";
                }
                ss << (uint64_t)mData[k] << ",";
            }
            ss << "}\n";
            ss << "{valid : \n";
            ss << std::hex;
            for (int k = 0; k < mValid.size(); ++k) {
                ss << "0x" << mValid[k] << ",";
                if (!(k % size) && (k != 0)) {
                    ss << "\n";
                }
            }
            ss << "}}\n";
            return ss.str();
        }

    private:
        Vector& emplace() {
            if (mDataState == State::GpuAlloc) {
                mDataState = State::Gpu;
                mColumn.size = mData.size();
                mColumn.dtype = GdfDataType<Type>::Value;
                cudaMalloc((void**)&(mColumn.data), mData.size() * sizeof(Type));
            }
            if (mDataState == State::GpuCopy) {
                mDataState = State::Gpu;
                mColumn.size = mData.size();
                mColumn.dtype = GdfDataType<Type>::Value;
                cudaMalloc((void**)&(mColumn.data), mData.size() * sizeof(Type));
                cudaMemcpy(mColumn.data, mData.data(), mData.size() * sizeof(Type), cudaMemcpyHostToDevice);
            }
            if (mValidState == State::GpuAlloc) {
                mValidState = State::Gpu;
                cudaMalloc((void**)&(mColumn.valid), mValid.size() * sizeof(ValidType));
            }
            if (mValidState == State::GpuCopy) {
                mValidState = State::Gpu;
                cudaMalloc((void**)&(mColumn.valid), mValid.size() * sizeof(ValidType));
                cudaMemcpy(mColumn.valid, mValid.data(), mValid.size() * sizeof(ValidType), cudaMemcpyHostToDevice);
            }
            return *this;
        }

    private:
        std::vector<Type> mData;
        std::vector<ValidType> mValid;

    private:
        gdf_column mColumn;

    private:
        State mDataState = State::Init;
        State mValidState = State::Init;
    };
}
}

#endif
