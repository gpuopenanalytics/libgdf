#ifndef GDF_LIBRARY_VECTOR_H
#define GDF_LIBRARY_VECTOR_H

#include "gdf/gdf.h"
#include "library/types.h"
#include "library/field.h"

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
            mData.clear();
            mValid.clear();
        }

        Vector& clear() {
            eraseGpu();
            return *this;
        }

        Vector& range(Type init, Type final, Type step) {
            assert((Type)0 < step);
            assert(init < final);

            int size = (final - init) / step;
            mData.resize(size);
            for (int k = 0; k < size; ++k) {
                mData[k] = init;
                init += step;
            }
            mData.write();
            updateData();
            return *this;
        }

        Vector& fill(int size, Type value) {
            mData.resize(size);
            std::fill(mData.begin(), mData.end(), value);
            mData.write();
            updateData();
            return *this;
        }

        Vector& valid(bool value) {
            int size = (mData.size() / ValidSize) + ((mData.size() % ValidSize) ? 1 : 0);
            mValid.resize(size);
            std::generate(mValid.begin(), mValid.end(), [value] { return -(ValidType)value; });
            mValid.write();
            updateValid();
            return *this;
        }

        // TODO: implementation incomplete
        Vector& valid(bool value, std::initializer_list<int> list) {
            return *this;
        }

        Vector& valid(bool value, int init, int final, int step) {
            int size = (mData.size() / ValidSize) + ((mData.size() % ValidSize) ? 1 : 0);
            mValid.resize(size);
            for (int index = 0; index < size; ++index) {
                ValidType val = 0;
                while (((init / ValidSize) == index) && (init < final)) {
                    val |= (1 << (init % ValidSize));
                    init += step;
                }
                if (value) {
                    mValid[index] = val;
                } else {
                    mValid[index] = ~val;
                }
            }
            mValid.write();
            updateValid();
            return *this;
        }

        void emplace(int size) {
            int validSize = (size / ValidSize) + ((size % ValidSize) ? 1 : 0);
            mData.resize(size);
            mValid.resize(validSize);
            updateData();
            updateValid();
        }

        void read() {
            mData.read();
            mValid.read();
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
        void updateData() {
            mColumn.size = mData.size();
            mColumn.dtype = GdfDataType<Type>::Value;
            mColumn.data = (void*)mData.getGpuData();
        }

        void updateValid() {
            mColumn.valid = (gdf_valid_type*)mValid.getGpuData();
        }

    private:
        gdf_column mColumn;

    private:
        Field<Type> mData;
        Field<ValidType> mValid;
    };
}
}

#endif
