// main_omp.cpp -- OpenMP version. Same kernel (fire.h), same output hash as serial.
// usage: OMP_NUM_THREADS=8 ./omp [rows] [cols] [N] [tmax] [seed] [stencil|ensemble]
//   stencil  : parallelise the cell sweep inside one realisation (fine-grained)
//   ensemble : parallelise across realisations, one buffer set per thread (coarse-grained)
// Compare schedules without recompiling: OMP_SCHEDULE="static" / "dynamic,64" / "guided"
#include "fire.h"
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

static uint32_t simulate_one(const LandView& L, const Scenario& S, uint64_t seed, uint32_t k,
                             uint32_t tmax, uint8_t* cur, uint8_t* nxt, uint16_t* res,
                             uint32_t* hits, bool inner_parallel) {
    const size_t n = (size_t)L.rows * L.cols;
    // First-touch happens here too: threads initialise the rows they will later update.
    #pragma omp parallel for if(inner_parallel) schedule(static)
    for (long c = 0; c < (long)n; ++c) { cur[c] = L.fuel[c] ? UNBURNED : UNBURNABLE; res[c] = 0; }

    RealParams rp = sample_realisation(L, S, seed, k);
    if (rp.ig_i < 0) return 0;
    size_t ig = (size_t)rp.ig_i * L.cols + rp.ig_j;
    cur[ig] = BURNING; res[ig] = fuel_model(L.fuel[ig]).residence;

    uint32_t t = 0;
    size_t burning = 1;
    while (burning > 0 && t < tmax) {
        burning = 0;
        // nxt[] is written only at the cell's own index, so no critical sections or atomics.
        #pragma omp parallel for collapse(2) reduction(+:burning) schedule(runtime) if(inner_parallel)
        for (int i = 0; i < L.rows; ++i)
            for (int j = 0; j < L.cols; ++j) {
                update_cell(L, rp, seed, k, t, i, j, cur, nxt, res);
                burning += (nxt[(size_t)i * L.cols + j] == BURNING);
            }
        std::swap(cur, nxt);
        ++t;
    }
    #pragma omp parallel for if(inner_parallel) schedule(static)
    for (long c = 0; c < (long)n; ++c) hits[c] += (cur[c] == BURNED || cur[c] == BURNING);
    return t;
}

static uint64_t fnv1a(const void* p, size_t bytes) {
    const uint8_t* b = (const uint8_t*)p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    // Default to static scheduling: the runtime default of libgomp is dynamic,1, which made
    // the cell sweep up to ~10x slower. An explicit OMP_SCHEDULE from the user still wins.
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_static, 0);
    int rows = argc > 1 ? atoi(argv[1]) : 512;
    int cols = argc > 2 ? atoi(argv[2]) : rows;
    int N    = argc > 3 ? atoi(argv[3]) : 20;
    uint32_t tmax = argc > 4 ? (uint32_t)atoi(argv[4]) : 2000;
    uint64_t seed = argc > 5 ? strtoull(argv[5], nullptr, 10) : 42;
    bool ensemble = argc > 6 && strcmp(argv[6], "ensemble") == 0;

    Landscape land = make_landscape(rows, cols, seed);
    LandView L = land.view();
    Scenario S = {8.f, 3.f, 0.785f, 0.5f, 0.08f, 0.03f};
    const size_t n = (size_t)rows * cols;
    std::vector<uint32_t> hits(n, 0);
    uint64_t total_steps = 0;

    double t0 = omp_get_wtime();
    if (!ensemble) {
        std::vector<uint8_t> a(n), b(n); std::vector<uint16_t> res(n);
        for (int k = 0; k < N; ++k)
            total_steps += simulate_one(L, S, seed, (uint32_t)k, tmax, a.data(), b.data(),
                                        res.data(), hits.data(), true);
    } else {
        #pragma omp parallel reduction(+:total_steps)
        {
            std::vector<uint8_t> a(n), b(n); std::vector<uint16_t> res(n);
            std::vector<uint32_t> local(n, 0);                 // thread-private accumulator
            // realisation cost varies a lot -> dynamic, chunk 1
            #pragma omp for schedule(dynamic, 1)
            for (int k = 0; k < N; ++k)
                total_steps += simulate_one(L, S, seed, (uint32_t)k, tmax, a.data(), b.data(),
                                            res.data(), local.data(), false);
            #pragma omp critical
            for (size_t c = 0; c < n; ++c) hits[c] += local[c];   // integer sum: order-independent
        }
    }
    double sec = omp_get_wtime() - t0;

    double bp_sum = 0; uint32_t bp_max = 0;
    for (size_t c = 0; c < n; ++c) { bp_sum += hits[c]; bp_max = std::max(bp_max, hits[c]); }
    printf("grid %dx%d  N=%d  tmax=%u  seed=%llu  mode=%s  threads=%d\n", rows, cols, N, tmax,
           (unsigned long long)seed, ensemble ? "ensemble" : "stencil", omp_get_max_threads());
    printf("time %.3f s   steps %llu   MCUPS %.1f\n", sec, (unsigned long long)total_steps,
           (double)n * total_steps / (sec * 1e6));
    printf("mean BP %.4f   max BP %.3f\n", bp_sum / ((double)n * N), (double)bp_max / N);
    printf("HASH %016llx\n", (unsigned long long)fnv1a(hits.data(), n * sizeof(uint32_t)));
    return 0;
}
