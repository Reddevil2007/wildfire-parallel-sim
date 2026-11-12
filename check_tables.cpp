// check_tables.cpp -- runs the CUDA version's table-driven kernel (fire_tables.h) on the CPU.
// Its HASH must equal ./serial's for the same arguments: that shows the GPU path computes
// exactly the same model, without needing a GPU.
// usage: ./check_tables [rows] [cols] [N] [tmax] [seed]
#include "fire_tables.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

static uint32_t simulate_one(const LandView& L, const float* ps, const Scenario& S, uint64_t seed,
                             uint32_t k, uint32_t tmax, uint8_t* cur, uint8_t* nxt, uint16_t* res,
                             uint32_t* hits) {
    size_t n = (size_t)L.rows * L.cols;
    for (size_t c = 0; c < n; ++c) { cur[c] = L.fuel[c] ? UNBURNED : UNBURNABLE; res[c] = 0; }
    RealParams rp = sample_realisation(L, S, seed, k);
    if (rp.ig_i < 0) return 0;
    StepParams sp = make_step_params(rp);
    size_t ig = (size_t)rp.ig_i * L.cols + rp.ig_j;
    cur[ig] = BURNING; res[ig] = fuel_model(L.fuel[ig]).residence;

    uint32_t t = 0;
    size_t burning = 1;
    while (burning > 0 && t < tmax) {
        burning = 0;
        for (int i = 0; i < L.rows; ++i)
            for (int j = 0; j < L.cols; ++j) {
                update_cell_tab(L.rows, L.cols, L.fuel, ps, sp, seed, k, t, i, j, cur, nxt, res);
                burning += (nxt[(size_t)i * L.cols + j] == BURNING);
            }
        std::swap(cur, nxt);
        ++t;
    }
    for (size_t c = 0; c < n; ++c) hits[c] += (cur[c] == BURNED || cur[c] == BURNING);
    return t;
}

static uint64_t fnv1a(const void* p, size_t bytes) {
    const uint8_t* b = (const uint8_t*)p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 512;
    int cols = argc > 2 ? atoi(argv[2]) : rows;
    int N    = argc > 3 ? atoi(argv[3]) : 20;
    uint32_t tmax = argc > 4 ? (uint32_t)atoi(argv[4]) : 2000;
    uint64_t seed = argc > 5 ? strtoull(argv[5], nullptr, 10) : 42;

    Landscape land = make_landscape(rows, cols, seed);
    LandView L = land.view();
    Scenario S = {8.f, 3.f, 0.785f, 0.5f, 0.08f, 0.03f};
    std::vector<float> ps = make_slope_table(L);

    size_t n = (size_t)rows * cols;
    std::vector<uint8_t> a(n), b(n); std::vector<uint16_t> res(n); std::vector<uint32_t> hits(n, 0);
    uint64_t total_steps = 0;
    for (int k = 0; k < N; ++k)
        total_steps += simulate_one(L, ps.data(), S, seed, (uint32_t)k, tmax, a.data(), b.data(),
                                    res.data(), hits.data());

    printf("grid %dx%d  N=%d  tmax=%u  seed=%llu  (table kernel on CPU)\n", rows, cols, N, tmax,
           (unsigned long long)seed);
    printf("steps %llu\n", (unsigned long long)total_steps);
    printf("HASH %016llx\n", (unsigned long long)fnv1a(hits.data(), n * sizeof(uint32_t)));
    return 0;
}
