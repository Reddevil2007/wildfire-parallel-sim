// fire.h -- shared kernel interface for the PCAP wildfire project.
// Everything the serial, OpenMP, MPI and CUDA versions must agree on lives here.
// Rules: raw pointers + POD structs only in the kernel (so CUDA can reuse it),
// integer-only RNG (so CPU and GPU draw identical numbers), gather-style update
// (a cell writes only itself, so no write races in any backend).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>

#ifdef __CUDACC__
#define FIRE_HD __host__ __device__
#else
#define FIRE_HD
#endif

// ---------------------------------------------------------------- states
enum : uint8_t { UNBURNABLE = 0, UNBURNED = 1, BURNING = 2, BURNED = 3 };

// ---------------------------------------------------------------- fuel table
// Simplified stand-in for Rothermel: p_h is a per-step base ignition probability
// (NOT yet calibrated against R(theta); see TODO in the report), residence is
// the number of steps a cell stays BURNING.
struct FuelModel { float p_h, p_veg, p_den; uint16_t residence; };

FIRE_HD inline FuelModel fuel_model(int c) {
    switch (c) {
        case 1:  return {0.30f, 0.0f, 0.0f, 3};    // grass
        case 2:  return {0.25f, 0.2f, 0.1f, 6};    // shrub
        case 3:  return {0.20f, 0.4f, 0.4f, 12};   // forest
        default: return {0.0f, 0.0f, 0.0f, 0};     // unburnable
    }
}

// Alexandridis et al. (2008) coefficients; moisture extinction fraction.
static constexpr float C1 = 0.045f, C2 = 0.131f, A_SLOPE = 0.078f, M_EXT = 0.30f;

// ---------------------------------------------------------------- data views
struct LandView {                       // POD, safe to pass to a CUDA kernel
    int rows, cols;
    float cell_m;                       // cell size in metres
    const uint8_t* fuel;                // 0 = unburnable
    const float* elev;                  // metres
};

struct Scenario {                       // distributions of the uncertain inputs
    float wind_mean, wind_sd;           // m/s
    float dir_mean, dir_sd;             // radians, direction wind blows TOWARD (x=col, y=row)
    float moist_mean, moist_sd;         // dead fuel moisture fraction
};

struct RealParams {                     // one Monte Carlo realisation
    float wind, wind_cos, wind_sin, moisture;
    int ig_i, ig_j;                     // ignition cell; -1 if no burnable cell
};

// ---------------------------------------------------------------- RNG
// Counter-based: output depends only on (seed, k, cell, ctr), never on thread,
// rank or block count. Integer arithmetic only => bitwise identical on CPU/GPU.
FIRE_HD inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
// uniform in (0,1): 23 bits so the float is exact
FIRE_HD inline float rng_u01(uint64_t seed, uint32_t k, uint32_t cell, uint32_t ctr) {
    uint64_t x = mix64(seed ^ mix64((uint64_t)k ^ mix64(((uint64_t)cell << 32) | ctr)));
    return (float)((x >> 41) + 0.5) * (1.0f / 8388608.0f);
}

static constexpr uint32_t SAMPLE_CELL = 0xFFFFFFFFu;   // reserved "cell" id for per-realisation draws

FIRE_HD inline float rng_normal(uint64_t seed, uint32_t k, uint32_t ctr) {
    float u1 = rng_u01(seed, k, SAMPLE_CELL, ctr), u2 = rng_u01(seed, k, SAMPLE_CELL, ctr + 1);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);   // Box-Muller
}

// Draw realisation k. Direction uses a wrapped normal (approximation to von Mises).
FIRE_HD inline RealParams sample_realisation(const LandView& L, const Scenario& S,
                                             uint64_t seed, uint32_t k) {
    RealParams r;
    float w = S.wind_mean + S.wind_sd * rng_normal(seed, k, 0);
    r.wind = w < 0.f ? 0.f : w;
    float d = S.dir_mean + S.dir_sd * rng_normal(seed, k, 2);
    r.wind_cos = cosf(d); r.wind_sin = sinf(d);
    float m = S.moist_mean + S.moist_sd * rng_normal(seed, k, 4);
    r.moisture = m < 0.f ? 0.f : (m > M_EXT ? M_EXT : m);
    // ignition: uniform cell, then scan forward (wrapping) to the next burnable one
    size_t n = (size_t)L.rows * L.cols;
    size_t start = (size_t)(rng_u01(seed, k, SAMPLE_CELL, 6) * (float)n);
    if (start >= n) start = n - 1;
    r.ig_i = r.ig_j = -1;
    for (size_t s = 0; s < n; ++s) {
        size_t id = (start + s) % n;
        if (L.fuel[id] != 0) { r.ig_i = (int)(id / L.cols); r.ig_j = (int)(id % L.cols); break; }
    }
    return r;
}

// ---------------------------------------------------------------- the kernel
// Advance ONE cell by one step. Reads cur[] of the 8 neighbours, writes only
// nxt[id] and res[id] of its own cell. This is the function OpenMP loops over and
// the CUDA kernel calls per thread.
FIRE_HD inline void update_cell(const LandView& L, const RealParams& rp, uint64_t seed,
                                uint32_t k, uint32_t t, int i, int j,
                                const uint8_t* cur, uint8_t* nxt, uint16_t* res) {
    size_t id = (size_t)i * L.cols + j;
    uint8_t s = cur[id];
    if (s == BURNING) {
        if (res[id] <= 1) { nxt[id] = BURNED; res[id] = 0; }
        else              { nxt[id] = BURNING; res[id]--; }
        return;
    }
    nxt[id] = s;
    if (s != UNBURNED) return;

    FuelModel fm = fuel_model(L.fuel[id]);
    float p_m = 1.0f - rp.moisture / M_EXT;
    if (p_m <= 0.f) return;

    for (int d = 0; d < 9; ++d) {
        if (d == 4) continue;
        int dr = d / 3 - 1, dc = d % 3 - 1;          // neighbour offset (row, col)
        int ni = i + dr, nj = j + dc;
        if (ni < 0 || ni >= L.rows || nj < 0 || nj >= L.cols) continue;
        size_t nid = (size_t)ni * L.cols + nj;
        if (cur[nid] != BURNING) continue;

        float dist = (dr != 0 && dc != 0) ? 1.41421356f : 1.0f;
        float dx = -dc / dist, dy = -dr / dist;      // direction fire travels: neighbour -> me
        float cosang = dx * rp.wind_cos + dy * rp.wind_sin;
        float p_w = expf(C1 * rp.wind) * expf(C2 * rp.wind * (cosang - 1.0f));
        float theta = atanf((L.elev[id] - L.elev[nid]) / (dist * L.cell_m)) * 57.29578f; // deg, + = uphill
        float p_s = expf(A_SLOPE * theta);
        float p = fm.p_h * (1.f + fm.p_veg) * (1.f + fm.p_den) * p_w * p_s * p_m;
        if (p > 1.f) p = 1.f;
        if (rng_u01(seed, k, (uint32_t)id, t * 8u + (uint32_t)(d > 4 ? d - 1 : d)) < p) {
            nxt[id] = BURNING; res[id] = fm.residence;
            return;
        }
    }
}

// ---------------------------------------------------------------- owning landscape (host only)
struct Landscape {
    int rows = 0, cols = 0; float cell_m = 30.f;
    std::vector<uint8_t> fuel; std::vector<float> elev;
    LandView view() const { return {rows, cols, cell_m, fuel.data(), elev.data()}; }
};

// Synthetic terrain + fuel: smooth hills, 16x16-cell fuel blocks (10% unburnable).
inline Landscape make_landscape(int rows, int cols, uint64_t seed) {
    Landscape L; L.rows = rows; L.cols = cols;
    L.fuel.resize((size_t)rows * cols); L.elev.resize((size_t)rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            size_t id = (size_t)i * cols + j;
            L.elev[id] = 200.f * sinf(i / 97.f) * cosf(j / 131.f) + 100.f * sinf((i + j) / 53.f);
            uint64_t h = mix64(seed ^ ((uint64_t)(i / 16) << 32 | (uint64_t)(j / 16))) % 10;
            L.fuel[id] = h == 0 ? 0 : (h <= 4 ? 1 : (h <= 7 ? 2 : 3));
        }
    return L;
}
