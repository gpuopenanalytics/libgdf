#include "gtest/gtest.h"
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
    void assertVector(gdf::library::Vector<TypeOut>& out,
                      gdf::library::Vector<TypeVax>& vax,
                      gdf::library::Scalar<TypeVay>& vay,
                      TypeOpe&& ope) {
        ASSERT_TRUE(out.dataSize() == vax.dataSize());
        for (int index = 0; index < out.dataSize(); ++index) {
            ASSERT_TRUE(out.data(index) == (TypeOut)(ope(vax.data(index), gdf::library::getScalar(TypeVay{}, vay.scalar()))));
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    void assertVector(gdf::library::Vector<TypeOut>& out,
                      gdf::library::Vector<TypeVax>& vax,
                      gdf::library::Vector<TypeVay>& vay,
                      TypeOpe&& ope) {
        ASSERT_TRUE(out.dataSize() == vax.dataSize());
        ASSERT_TRUE(out.dataSize() == vay.dataSize());
        for (int index = 0; index < out.dataSize(); ++index) {
            ASSERT_TRUE(out.data(index) == (TypeOut)(ope(vax.data(index), vay.data(index))));
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeDef, typename TypeOpe>
    void assertVector(gdf::library::Vector<TypeOut>& out,
                      gdf::library::Vector<TypeVax>& vax,
                      gdf::library::Scalar<TypeVay>& vay,
                      gdf::library::Scalar<TypeDef>& def,
                      TypeOpe&& ope) {
        using ValidType = typename gdf::library::Vector<TypeOut>::ValidType;
        int ValidSize = gdf::library::Vector<TypeOut>::ValidSize;

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
                vax_aux = (TypeVax) gdf::library::getScalar(TypeDef{}, def.scalar());
            }

            ASSERT_TRUE((out.valid(k) & mask) == mask);
            ASSERT_TRUE(out.data(index) == (TypeOut)(ope(vax_aux, (TypeVay)vay)));
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeDef, typename TypeOpe>
    void assertVector(gdf::library::Vector<TypeOut>& out,
                      gdf::library::Vector<TypeVax>& vax,
                      gdf::library::Vector<TypeVay>& vay,
                      gdf::library::Scalar<TypeDef>& def,
                      TypeOpe&& ope) {
        using ValidType = typename gdf::library::Vector<TypeOut>::ValidType;
        int ValidSize = gdf::library::Vector<TypeOut>::ValidSize;

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
                vax_aux = (TypeVax) gdf::library::getScalar(TypeDef{}, def.scalar());
            }

            TypeVay vay_aux = vay.data(index);
            ValidType vay_val = vay.valid(k) & mask;
            if (vay_val == 0) {
                vay_aux = (TypeVay) gdf::library::getScalar(TypeDef{}, def.scalar());
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
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using ADD = gdf::library::operation::Add<SI32, FP32, UI32>;

    gdf::library::Vector<SI32> out;
    gdf::library::Scalar<FP32> vax;
    gdf::library::Vector<UI32> vay;

    vay.range(0, 32, 1);
    vax.set(100);
    out.emplace(vay.dataSize());

    gdf_binary_operation_v_s_v(out.column(), vax.scalar(), vay.column(), GDF_ADD);

    out.read();

    assertVector(out, vay, vax, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_UI16_SI16) {
    using SI08 = gdf::library::GdfEnumType<GDF_INT8>;
    using UI16 = gdf::library::GdfEnumType<GDF_UINT16>;
    using SI16 = gdf::library::GdfEnumType<GDF_INT16>;
    using ADD = gdf::library::operation::Add<SI08, UI16, SI16>;

    gdf::library::Vector<SI08> out;
    gdf::library::Vector<UI16> vax;
    gdf::library::Scalar<SI16> vay;

    vax.range(0, 32, 1);
    vay.set(100);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_s(out.column(), vax.column(), vay.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_UI32_FP64_SI08) {
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using SI08 = gdf::library::GdfEnumType<GDF_INT8>;
    using ADD = gdf::library::operation::Add<UI32, FP64, SI08>;

    gdf::library::Vector<UI32> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Vector<SI08> vay;

    vax.range(0, 30, 1);
    vay.range(0, 60, 2);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_Default_SI32_SI16_UI64_SI64) {
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using SI16 = gdf::library::GdfEnumType<GDF_INT16>;
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using SI64 = gdf::library::GdfEnumType<GDF_INT64>;
    using ADD = gdf::library::operation::Add<SI32, SI16, UI64>;

    gdf::library::Vector<SI32> out;
    gdf::library::Scalar<SI16> vax;
    gdf::library::Vector<UI64> vay;
    gdf::library::Scalar<SI64> def;

    vax.set(50);
    vay.range(0, 32, 1).valid(false, 0, 32, 4);
    def.set(100);
    out.emplace(vay.dataSize());

    gdf_binary_operation_v_s_v_d(out.column(), vax.scalar(), vay.column(), def.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vay, vax, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_Default_FP32_SI16_UI08_UI32) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using SI16 = gdf::library::GdfEnumType<GDF_INT16>;
    using UI08 = gdf::library::GdfEnumType<GDF_UINT8>;
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using ADD = gdf::library::operation::Add<FP32, SI16, UI08>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<SI16> vax;
    gdf::library::Scalar<UI08> vay;
    gdf::library::Scalar<UI32> def;

    vax.range(0, 96, 3).valid(false, 0, 32, 4);
    vay.set(50);
    def.set(150);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_Default_FP64_SI32_UI32_UI16) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using UI16 = gdf::library::GdfEnumType<GDF_UINT16>;
    using ADD = gdf::library::operation::Add<FP64, SI32, UI32>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<SI32> vax;
    gdf::library::Vector<UI32> vay;
    gdf::library::Scalar<UI16> def;

    vax.range(0, 32, 1).valid(false, 0, 32, 3);
    vay.range(0, 64, 2).valid(false, 0, 32, 4);
    def.set(150);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);

    out.read();

    assertVector(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using SUB = gdf::library::operation::Sub<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.range(100, 200, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_SUB);

    out.read();

    assertVector(out, vax, vay, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using MUL = gdf::library::operation::Mul<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.range(100, 200, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MUL);

    out.read();

    assertVector(out, vax, vay, MUL());
}


TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using DIV = gdf::library::operation::Div<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.range(100, 200, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_DIV);

    out.read();

    assertVector(out, vax, vay, DIV());
}


TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using TRUEDIV = gdf::library::operation::TrueDiv<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.range(100, 200, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_TRUE_DIV);

    out.read();

    assertVector(out, vax, vay, TRUEDIV());
}


TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using FLOORDIV = gdf::library::operation::FloorDiv<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.range(100, 200, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_FLOOR_DIV);

    out.read();

    assertVector(out, vax, vay, FLOORDIV());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using MOD = gdf::library::operation::Mod<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.range(120, 220, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MOD);

    out.read();

    assertVector(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using MOD = gdf::library::operation::Mod<FP32, FP32, FP32>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<FP32> vax;
    gdf::library::Vector<FP32> vay;

    vax.range(120, 220, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MOD);

    out.read();

    assertVector(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using MOD = gdf::library::operation::Mod<FP64, FP64, FP64>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Vector<FP64> vay;

    vax.range(120, 220, 2);
    vay.range(50, 100, 1);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MOD);

    out.read();

    assertVector(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using POW = gdf::library::operation::Pow<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.range(0, 50, 1);
    vay.fill(50, 2);
    out.emplace(vax.dataSize());

    gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_POW);

    out.read();

    assertVector(out, vax, vay, POW());
}
