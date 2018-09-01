#include <chrono>
#include <thread>
#include <cstdint>
#include "benchmark/benchmark.h"
#include "library/scalar.h"
#include "library/vector.h"

namespace bench {
    template <typename Type>
    struct BinaryOperation : public benchmark::Fixture {
    public:
        virtual void SetUp(benchmark::State& state) {
            if (id != state.range(0)) {
                id = state.range(0);
                sampler.counter = 0;
                sampler.limit = state.range(1);
                sampler.data.clear();
            }

            vax.range((Type)state.range(2), (Type)state.range(3), (Type)state.range(4));
            vax.valid(true, (int)state.range(2), (int)state.range(3), 3);

            vay.range((Type)state.range(2), (Type)state.range(3), (Type)state.range(4));
            vay.valid(true, (int)state.range(2), (int)state.range(3), 4);

            out.emplace(vax.dataSize());
            def.set((Type)state.range(3));
        }

        virtual void TearDown(benchmark::State& state) {
            vax.clear();
            vay.clear();
            out.clear();
        }

        void startTime() {
            startPointClock = std::chrono::high_resolution_clock::now();
        }

        void stopTime() {
            stopPointClock = std::chrono::high_resolution_clock::now();
        }

        void saveTime(benchmark::State& state) {
            auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(stopPointClock - startPointClock);
            state.SetIterationTime(elapsedTimeSeconds.count());
        }

        void makeSamplerData(benchmark::State& state) {
            if (sampler.counter < sampler.limit) {
                auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(stopPointClock - startPointClock);
                sampler.data.push_back(elapsedTimeSeconds.count());
                sampler.counter++;
            }
        }

        void saveSamplerData(benchmark::State& state) {
            for (int k = 0; k < sampler.data.size(); ++k) {
                state.counters.insert({sampler.name + std::to_string(k), {sampler.data[k]}});
            }
        }

        gdf::library::Vector<Type> out;
        gdf::library::Vector<Type> vax;
        gdf::library::Vector<Type> vay;
        gdf::library::Scalar<Type> def;

        std::chrono::time_point<std::chrono::high_resolution_clock> startPointClock;
        std::chrono::time_point<std::chrono::high_resolution_clock> stopPointClock;

        struct Sampler {
            int limit = 0;
            int counter = 0;
            std::vector<double> data;
            const std::string name { "Iter" };
        };

        int id = 0;
        Sampler sampler;
    };

    BENCHMARK_TEMPLATE_DEFINE_F(BinaryOperation, KernelInteger, uint64_t)(benchmark::State& state) {
        for (auto _ : state) {
            startTime();
            gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);
            stopTime();
            saveTime(state);
            makeSamplerData(state);
        }
        saveSamplerData(state);
    }

    BENCHMARK_TEMPLATE_DEFINE_F(BinaryOperation, KernelFloat, float)(benchmark::State& state) {
        for (auto _ : state) {
            startTime();
            gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);
            stopTime();
            saveTime(state);
            makeSamplerData(state);
        }
        saveSamplerData(state);
    }

    BENCHMARK_TEMPLATE_DEFINE_F(BinaryOperation, KernelDouble, double)(benchmark::State& state) {
        for (auto _ : state) {
            startTime();
            gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);
            stopTime();
            saveTime(state);
            makeSamplerData(state);
        }
        saveSamplerData(state);
    }

    BENCHMARK_TEMPLATE_DEFINE_F(BinaryOperation, KernelIntegerDefault, uint64_t)(benchmark::State& state) {
        for (auto _ : state) {
            startTime();
            gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
            stopTime();
            saveTime(state);
            makeSamplerData(state);
        }
        saveSamplerData(state);
    }

    BENCHMARK_TEMPLATE_DEFINE_F(BinaryOperation, KernelFloatDefault, float)(benchmark::State& state) {
        for (auto _ : state) {
            startTime();
            gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
            stopTime();
            saveTime(state);
            makeSamplerData(state);
        }
        saveSamplerData(state);
    }

    BENCHMARK_TEMPLATE_DEFINE_F(BinaryOperation, KernelDoubleDefault, double)(benchmark::State& state) {
        for (auto _ : state) {
            startTime();
            gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
            stopTime();
            saveTime(state);
            makeSamplerData(state);
        }
        saveSamplerData(state);
    }

    BENCHMARK_REGISTER_F(BinaryOperation, KernelInteger)
        ->UseManualTime()
        ->Args({1, 7, 0, 10, 1})
        ->Args({2, 7, 0, 1000, 1})
        ->Args({3, 7, 0, 100000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelFloat)
        ->UseManualTime()
        ->Args({1, 7, 0, 10, 1})
        ->Args({2, 7, 0, 1000, 1})
        ->Args({3, 7, 0, 100000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelDouble)
        ->UseManualTime()
        ->Args({1, 7, 0, 10, 1})
        ->Args({2, 7, 0, 1000, 1})
        ->Args({3, 7, 0, 100000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelIntegerDefault)
        ->UseManualTime()
        ->Args({1, 7, 0, 10, 1})
        ->Args({2, 7, 0, 1000, 1})
        ->Args({3, 7, 0, 100000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelFloatDefault)
        ->UseManualTime()
        ->Args({1, 7, 0, 10, 1})
        ->Args({2, 7, 0, 1000, 1})
        ->Args({3, 7, 0, 100000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelDoubleDefault)
        ->UseManualTime()
        ->Args({1, 7, 0, 10, 1})
        ->Args({2, 7, 0, 1000, 1})
        ->Args({3, 7, 0, 100000, 1});
}
