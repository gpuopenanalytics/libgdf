#ifndef GDF_LIBRARY_FIELD_H
#define GDF_LIBRARY_FIELD_H

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace gdf {
namespace library {

    template <typename Type>
    class Field {
    public:
        ~Field() {
            destroy();
        }

    public:
        void clear() {
            mCpuData.clear();
            destroy();
        }

        void resize(int size) {
            int sizeBytes = size * sizeof(Type);
            if (sizeBytes != mSizeAllocBytes) {
                mCpuData.resize(size);
                destroy();
                create(size);
            }
        }

    public:
        auto getGpuData() -> Type* {
            return mGpuData;
        }

    public:
        auto begin() -> typename std::vector<Type>::iterator {
            return mCpuData.begin();
        }

        auto end() -> typename std::vector<Type>::iterator {
            return mCpuData.end();
        }

    public:
        auto size() -> std::size_t {
            return mCpuData.size();
        }

        auto operator[](int index) -> Type& {
            assert(mCpuData.size() < index);
            return mCpuData[index];
        }

    public:
        void write() {
            if (mSizeAllocBytes) {
                cudaMemcpy(mGpuData, mCpuData.data(), mSizeAllocBytes, cudaMemcpyHostToDevice);
            }
        }

        void read() {
            if (mSizeAllocBytes) {
                cudaMemcpy(mCpuData.data(), mGpuData, mSizeAllocBytes, cudaMemcpyDeviceToHost);
            }
        }

    protected:
        void create(int size) {
            mSizeAllocBytes = size * sizeof(Type);
            cudaMalloc((void**)&(mGpuData), mSizeAllocBytes);
        }

        void destroy() {
            if (mSizeAllocBytes) {
                mSizeAllocBytes = 0;
                cudaFree(mGpuData);
            }
        }

    private:
        int mSizeAllocBytes {0};
        Type* mGpuData {nullptr};
        std::vector<Type> mCpuData;
    };

}
}

#endif
