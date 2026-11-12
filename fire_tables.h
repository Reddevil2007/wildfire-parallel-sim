// fire_tables.h -- table-driven form of update_cell() used by the CUDA version.
//
// Why: GPU expf/atanf are accurate but not bit-identical to the CPU's libm, so a
// GPU run of update_cell() could flip a rare "rng < p" test and drift from the
// serial result. Here every transcendental value is computed ONCE on the host
// (with the same expressions as fire.h) and the device only multiplies and
// compares, in the same order as update_cell(). IEEE multiply is exact-rounded on
// both, so the result is the same as the serial version.
//
// check_tables.cpp proves on the CPU that this path reproduces the serial hash.
#pragma once
#include "fire.h"

// Neighbour direction index: d = 0..8 without the centre (4) -> 0..7.
// Same mapping as the RNG counter offset in update_cell().
FIRE_HD inline int dir_index(int d) { return d > 4 ? d - 1 : d; }

struct StepParams {      // per realisation, filled on the host
    float p_w[8];        // wind factor for fire arriving from each neighbour direction
    float p_m;           // moisture factor
};

// Same expressions as the wind and moisture terms in update_cell().
inline StepParams make_step_params(const RealParams& rp) {
    StepParams sp;
    for (int d = 0; d < 9; ++d) {
        if (d == 4) continue;
        int dr = d / 3 - 1, dc = d % 3 - 1;
        float dist = (dr != 0 && dc != 0) ? 1.41421356f : 1.0f;
        float dx = -dc / dist, dy = -dr / dist;
        float cosang = dx * rp.wind_cos + dy * rp.wind_sin;
        sp.p_w[dir_index(d)] = expf(C1 * rp.wind) * expf(C2 * rp.wind * (cosang - 1.0f));
    }
    sp.p_m = 1.0f - rp.moisture / M_EXT;
    return sp;
}

// Slope factor for every (direction, cell): 8 x n floats, direction-major so
// neighbouring GPU threads read neighbouring addresses. Same expression as fire.h.
inline std::vector<float> make_slope_table(const LandView& L) {
    const size_t n = (size_t)L.rows * L.cols;
    std::vector<float> ps(8 * n, 0.0f);
    for (int i = 0; i < L.rows; ++i)
        for (int j = 0; j < L.cols; ++j) {
            size_t id = (size_t)i * L.cols + j;
            for (int d = 0; d < 9; ++d) {
                if (d == 4) continue;
                int dr = d / 3 - 1, dc = d % 3 - 1;
                int ni = i + dr, nj = j + dc;
                if (ni < 0 || ni >= L.rows || nj < 0 || nj >= L.cols) continue;
                size_t nid = (size_t)ni * L.cols + nj;
                float dist = (dr != 0 && dc != 0) ? 1.41421356f : 1.0f;
                float theta = atanf((L.elev[id] - L.elev[nid]) / (dist * L.cell_m)) * 57.29578f;
                ps[(size_t)dir_index(d) * n + id] = expf(A_SLOPE * theta);
            }
        }
    return ps;
}

// update_cell() with the wind, slope and moisture factors read from the tables.
// Reads cur[] of the 8 neighbours, writes only nxt[id] and res[id] of its own cell.
FIRE_HD inline void update_cell_tab(int rows, int cols, const uint8_t* fuel, const float* p_s,
                                    const StepParams& sp, uint64_t seed, uint32_t k, uint32_t t,
                                    int i, int j, const uint8_t* cur, uint8_t* nxt, uint16_t* res) {
    const size_t n = (size_t)rows * cols;
    size_t id = (size_t)i * cols + j;
    uint8_t s = cur[id];
    if (s == BURNING) {
        if (res[id] <= 1) { nxt[id] = BURNED; res[id] = 0; }
        else              { nxt[id] = BURNING; res[id]--; }
        return;
    }
    nxt[id] = s;
    if (s != UNBURNED) return;

    FuelModel fm = fuel_model(fuel[id]);
    float p_m = sp.p_m;
    if (p_m <= 0.f) return;

    for (int d = 0; d < 9; ++d) {
        if (d == 4) continue;
        int dr = d / 3 - 1, dc = d % 3 - 1;
        int ni = i + dr, nj = j + dc;
        if (ni < 0 || ni >= rows || nj < 0 || nj >= cols) continue;
        size_t nid = (size_t)ni * cols + nj;
        if (cur[nid] != BURNING) continue;

        int di = dir_index(d);
        float p = fm.p_h * (1.f + fm.p_veg) * (1.f + fm.p_den) * sp.p_w[di] * p_s[(size_t)di * n + id] * p_m;
        if (p > 1.f) p = 1.f;
        if (rng_u01(seed, k, (uint32_t)id, t * 8u + (uint32_t)di) < p) {
            nxt[id] = BURNING; res[id] = fm.residence;
            return;
        }
    }
}
