/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <thread>
#include <cstdint>
#include "benchmark/benchmark.h"
#include "tests/binary-operation/util/scalar.h"
#include "tests/binary-operation/util/vector.h"

namespace bench {
    static constexpr int NUMBER_COUNTERS = 6;

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

            vax.rangeData((Type)state.range(2), (Type)state.range(3), (Type)state.range(4));
            vax.rangeValid(true, (int)state.range(2), 3);

            vay.rangeData((Type)state.range(2), (Type)state.range(3), (Type)state.range(4));
            vay.rangeValid(true, (int)state.range(2), 4);

            out.emplaceVector(vax.dataSize());
            def.setValue((Type)state.range(3));
        }

        virtual void TearDown(benchmark::State& state) {
            vax.clearGpu();
            vay.clearGpu();
            out.clearGpu();
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
        ->Args({1, NUMBER_COUNTERS, 0, 100, 1})
        ->Args({2, NUMBER_COUNTERS, 0, 10000, 1})
        ->Args({3, NUMBER_COUNTERS, 0, 1000000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelFloat)
        ->UseManualTime()
        ->Args({1, NUMBER_COUNTERS, 0, 100, 1})
        ->Args({2, NUMBER_COUNTERS, 0, 10000, 1})
        ->Args({3, NUMBER_COUNTERS, 0, 1000000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelDouble)
        ->UseManualTime()
        ->Args({1, NUMBER_COUNTERS, 0, 100, 1})
        ->Args({2, NUMBER_COUNTERS, 0, 10000, 1})
        ->Args({3, NUMBER_COUNTERS, 0, 1000000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelIntegerDefault)
        ->UseManualTime()
        ->Args({1, NUMBER_COUNTERS, 0, 100, 1})
        ->Args({2, NUMBER_COUNTERS, 0, 10000, 1})
        ->Args({3, NUMBER_COUNTERS, 0, 1000000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelFloatDefault)
        ->UseManualTime()
        ->Args({1, NUMBER_COUNTERS, 0, 100, 1})
        ->Args({2, NUMBER_COUNTERS, 0, 10000, 1})
        ->Args({3, NUMBER_COUNTERS, 0, 1000000, 1});

    BENCHMARK_REGISTER_F(BinaryOperation, KernelDoubleDefault)
        ->UseManualTime()
        ->Args({1, NUMBER_COUNTERS, 0, 100, 1})
        ->Args({2, NUMBER_COUNTERS, 0, 10000, 1})
        ->Args({3, NUMBER_COUNTERS, 0, 1000000, 1});
}
