// main_serial.cpp -- serial reference. Every other backend must reproduce its output hash.
// usage: ./serial [rows] [cols] [N] [tmax] [seed]
#include "fire.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Run realisation k to extinction or tmax; add 1 to hits[] for every cell that burned.
// Returns number of steps taken.
static uint32_t simulate_one(const LandView& L, const Scenario& S, uint64_t seed, uint32_t k,
                             uint32_t tmax, uint8_t* cur, uint8_t* nxt, uint16_t* res, uint32_t* hits) {
    size_t n = (size_t)L.rows * L.cols;
    for (size_t c = 0; c < n; ++c) { cur[c] = L.fuel[c] ? UNBURNED : UNBURNABLE; res[c] = 0; }
    RealParams rp = sample_realisation(L, S, seed, k);
    if (rp.ig_i < 0) return 0;
    size_t ig = (size_t)rp.ig_i * L.cols + rp.ig_j;
    cur[ig] = BURNING; res[ig] = fuel_model(L.fuel[ig]).residence;

    uint32_t t = 0;
    size_t burning = 1;
    while (burning > 0 && t < tmax) {
        burning = 0;
        for (int i = 0; i < L.rows; ++i)
            for (int j = 0; j < L.cols; ++j) {
                update_cell(L, rp, seed, k, t, i, j, cur, nxt, res);
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

    size_t n = (size_t)rows * cols;
    std::vector<uint8_t> a(n), b(n); std::vector<uint16_t> res(n); std::vector<uint32_t> hits(n, 0);

    auto t0 = std::chrono::steady_clock::now();
    uint64_t total_steps = 0;
    for (int k = 0; k < N; ++k)
        total_steps += simulate_one(L, S, seed, (uint32_t)k, tmax, a.data(), b.data(), res.data(), hits.data());
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double bp_sum = 0; uint32_t bp_max = 0;
    for (size_t c = 0; c < n; ++c) { bp_sum += hits[c]; bp_max = std::max(bp_max, hits[c]); }
    printf("grid %dx%d  N=%d  tmax=%u  seed=%llu\n", rows, cols, N, tmax, (unsigned long long)seed);
    printf("time %.3f s   steps %llu   MCUPS %.1f\n", sec, (unsigned long long)total_steps,
           (double)n * total_steps / (sec * 1e6));
    printf("mean BP %.4f   max BP %.3f\n", bp_sum / ((double)n * N), (double)bp_max / N);
    printf("HASH %016llx\n", (unsigned long long)fnv1a(hits.data(), n * sizeof(uint32_t)));
    return 0;
}
