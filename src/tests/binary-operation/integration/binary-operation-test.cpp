#include <gtest/gtest.h>
#include "library/scalar.h"
#include "library/vector.h"
#include "library/operation.h"

struct BinaryOperationIntegrationTest : public ::testing::Test {
    BinaryOperationIntegrationTest() {
    }

    virtual ~BinaryOperationIntegrationTest() {
    }

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    void assertVector(gdf::test::Vector<TypeOut>& out,
                      gdf::test::Vector<TypeVax>& vax,
                      gdf::test::Scalar<TypeVay>& vay,
                      TypeOpe&& ope) {
        ASSERT_TRUE(out.dataSize() == vax.dataSize());
        for (int index = 0; index < out.dataSize(); ++index) {
            ASSERT_TRUE(out.data(index) == (TypeOut)(ope(vax.data(index), gdf::test::getScalar(TypeVay{}, vay.scalar()))));
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    void assertVector(gdf::test::Vector<TypeOut>& out,
                      gdf::test::Vector<TypeVax>& vax,
                      gdf::test::Vector<TypeVay>& vay,
                      TypeOpe&& ope) {
        ASSERT_TRUE(out.dataSize() == vax.dataSize());
        ASSERT_TRUE(out.dataSize() == vay.dataSize());
        for (int index = 0; index < out.dataSize(); ++index) {
            ASSERT_TRUE(out.data(index) == (TypeOut)(ope(vax.data(index), vay.data(index))));
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeDef, typename TypeOpe>
    void assertVector(gdf::test::Vector<TypeOut>& out,
                      gdf::test::Vector<TypeVax>& vax,
                      gdf::test::Scalar<TypeVay>& vay,
                      gdf::test::Scalar<TypeDef>& def,
                      TypeOpe&& ope) {
        using ValidType = typename gdf::test::Vector<TypeOut>::ValidType;
        int ValidSize = gdf::test::Vector<TypeOut>::ValidSize;

        ASSERT_TRUE(out.dataSize() == vax.dataSize());
        ASSERT_TRUE(out.validSize() == vax.validSize());

        ValidType mask = 1;
        for (int index = 0; index < out.dataSize(); ++index) {
            if (!(index % ValidSize)) {
                mask = 1;
            } else {
                mask <<= 1;
            }

            int k = index / ValidSize;

            TypeVax vax_aux = vax.data(index);
            ValidType vax_val = vax.valid(k) & mask;
            if (vax_val == 0) {
                vax_aux = (TypeVax) gdf::test::getScalar(TypeDef{}, def.scalar());
            }

            ASSERT_TRUE((out.valid(k) & mask) == mask);
            ASSERT_TRUE(out.data(index) == (TypeOut)(ope(vax_aux, (TypeVay)vay)));
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeDef, typename TypeOpe>
    void assertVector(gdf::test::Vector<TypeOut>& out,
                      gdf::test::Vector<TypeVax>& vax,
                      gdf::test::Vector<TypeVay>& vay,
                      gdf::test::Scalar<TypeDef>& def,
                      TypeOpe&& ope) {
        using ValidType = typename gdf::test::Vector<TypeOut>::ValidType;
        int ValidSize = gdf::test::Vector<TypeOut>::ValidSize;

        ASSERT_TRUE(out.dataSize() == vax.dataSize());
        ASSERT_TRUE(out.dataSize() == vay.dataSize());
        ASSERT_TRUE(out.validSize() == vax.validSize());
        ASSERT_TRUE(out.validSize() == vay.validSize());

        ValidType mask = 1;
        for (int index = 0; index < out.dataSize(); ++index) {
            if (!(index % ValidSize)) {
                mask = 1;
            } else {
                mask <<= 1;
            }

            int k = index / ValidSize;

            TypeVax vax_aux = vax.data(index);
            ValidType vax_val = vax.valid(k) & mask;
            if (vax_val == 0) {
                vax_aux = (TypeVax) gdf::test::getScalar(TypeDef{}, def.scalar());
            }

            TypeVay vay_aux = vay.data(index);
            ValidType vay_val = vay.valid(k) & mask;
            if (vay_val == 0) {
                vay_aux = (TypeVay) gdf::test::getScalar(TypeDef{}, def.scalar());
            }

            if ((vax_val | vay_val) == 0) {
                ASSERT_TRUE((out.valid(k) & mask) == 0);
            } else {
                ASSERT_TRUE((out.valid(k) & mask) == mask);
                ASSERT_TRUE(out.data(index) == (TypeOut)(ope(vax_aux, vay_aux)));
            }
        }
    }
};


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_SI32_FP32_UI32) {
    using SI32 = gdf::test::GdfEnumType<GDF_INT32>;
    using FP32 = gdf::test::GdfEnumType<GDF_FLOAT32>;
    using UI32 = gdf::test::GdfEnumType<GDF_UINT32>;
    using ADD = gdf::test::operation::Add<SI32, FP32, UI32>;

    gdf::test::Vector<SI32> out;
    gdf::test::Scalar<FP32> vax;
    gdf::test::Vector<UI32> vay;

    vay.range(0, 32, 1);
    vax.set(100);
    out.emplace(vay.dataSize());

    gdf_binary_operation_v_s_v(out.column(), vax.scalar(), vay.column(), GDF_ADD);

    out.read();

    assertVector(out, vay, vax, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_UI16_SI16) {
    using SI08 = gdf::test::GdfEnumType<GDF_INT8>;
    using UI16 = gdf::test::GdfEnumType<GDF_UINT16>;
    using SI16 = gdf::test::GdfEnumType<GDF_INT16>;
    using ADD = gdf::test::operation::Add<SI08, UI16, SI16>;

    gdf::test::Vector<SI08> out;
    gdf::test::Vector<UI16> vax;
    gdf::test::Scalar<SI16> vay;

    vax.range(0, 32, 1);
    vay.set(100);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_s(out.column(), vax.column(), vay.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_UI32_FP64_SI08) {
    using UI32 = gdf::test::GdfEnumType<GDF_UINT32>;
    using FP64 = gdf::test::GdfEnumType<GDF_FLOAT64>;
    using SI08 = gdf::test::GdfEnumType<GDF_INT8>;
    using ADD = gdf::test::operation::Add<UI32, FP64, SI08>;

    gdf::test::Vector<UI32> out;
    gdf::test::Vector<FP64> vax;
    gdf::test::Vector<SI08> vay;

    vax.range(0, 30, 1);
    vay.range(0, 60, 2);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_Default_SI32_SI16_UI64_SI64) {
    using SI32 = gdf::test::GdfEnumType<GDF_INT32>;
    using SI16 = gdf::test::GdfEnumType<GDF_INT16>;
    using UI64 = gdf::test::GdfEnumType<GDF_UINT64>;
    using SI64 = gdf::test::GdfEnumType<GDF_INT64>;
    using ADD = gdf::test::operation::Add<SI32, SI16, UI64>;

    gdf::test::Vector<SI32> out;
    gdf::test::Scalar<SI16> vax;
    gdf::test::Vector<UI64> vay;
    gdf::test::Scalar<SI64> def;

    vax.set(50);
    vay.range(0, 32, 1).valid(false, 0, 32, 4);
    def.set(100);
    out.emplace(vay.dataSize());

    gdf_binary_operation_v_s_v_d(out.column(), vax.scalar(), vay.column(), def.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vay, vax, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_Default_FP32_SI16_UI08_UI32) {
    using FP32 = gdf::test::GdfEnumType<GDF_FLOAT32>;
    using SI16 = gdf::test::GdfEnumType<GDF_INT16>;
    using UI08 = gdf::test::GdfEnumType<GDF_UINT8>;
    using UI32 = gdf::test::GdfEnumType<GDF_UINT32>;
    using ADD = gdf::test::operation::Add<FP32, SI16, UI08>;

    gdf::test::Vector<FP32> out;
    gdf::test::Vector<SI16> vax;
    gdf::test::Scalar<UI08> vay;
    gdf::test::Scalar<UI32> def;

    vax.range(0, 96, 3).valid(false, 0, 32, 4);
    vay.set(50);
    def.set(150);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_Default_FP64_SI32_UI32_UI16) {
    using FP64 = gdf::test::GdfEnumType<GDF_FLOAT64>;
    using SI32 = gdf::test::GdfEnumType<GDF_INT32>;
    using UI32 = gdf::test::GdfEnumType<GDF_UINT32>;
    using UI16 = gdf::test::GdfEnumType<GDF_UINT16>;
    using ADD = gdf::test::operation::Add<FP64, SI32, UI32>;

    gdf::test::Vector<FP64> out;
    gdf::test::Vector<SI32> vax;
    gdf::test::Vector<UI32> vay;
    gdf::test::Scalar<UI16> def;

    vax.range(0, 32, 1).valid(false, 0, 32, 3);
    vay.range(0, 64, 2).valid(false, 0, 32, 4);
    def.set(150);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, def, ADD());
}
