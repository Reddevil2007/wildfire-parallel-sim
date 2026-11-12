// main_cuda.cu -- CUDA version. Same model and same output format as main_serial.cpp.
// usage: ./cuda [rows] [cols] [N] [tmax] [seed]
//
// Both levels of parallelism run on the GPU at once:
//   * one thread per cell (fine-grained, the cell sweep), and
//   * one grid slice per realisation, blockIdx.z = k (coarse-grained, the ensemble).
// Every realisation advances together, one kernel launch per time step. A realisation
// that has burned out is only copied forward, so it costs almost nothing.
//
// The kernel calls update_cell_tab() from fire_tables.h, which takes the wind, slope and
// moisture factors from tables computed on the host. That keeps the GPU's floating-point
// results equal to the CPU's (see check_tables.cpp). Build with -fmad=false.
#include "fire_tables.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

static constexpr int BX = 32, BY = 8;   // 256 threads per block; x = columns for coalescing

// One time step for every cell of every realisation.
// cur/nxt/res hold N stacked grids of n cells; burning[k] counts BURNING cells in nxt.
__global__ void step_kernel(int rows, int cols, const uint8_t* __restrict__ fuel,
                            const float* __restrict__ p_s, const StepParams* __restrict__ sp,
                            const uint8_t* __restrict__ active, uint64_t seed, uint32_t t,
                            const uint8_t* __restrict__ cur, uint8_t* __restrict__ nxt,
                            uint16_t* __restrict__ res, unsigned int* __restrict__ burning) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t k = blockIdx.z;
    const size_t n = (size_t)rows * cols;
    const size_t off = (size_t)k * n;

    int is_burning = 0;
    if (i < rows && j < cols) {
        const size_t id = (size_t)i * cols + j;
        if (active[k]) {
            update_cell_tab(rows, cols, fuel, p_s, sp[k], seed, k, t, i, j,
                            cur + off, nxt + off, res + off);
            is_burning = (nxt[off + id] == BURNING);
        } else {
            nxt[off + id] = cur[off + id];       // burned out: just carry the state forward
        }
    }
    // Every thread reaches this line (no early return), so the block-wide count is safe.
    const int block_count = __syncthreads_count(is_burning);
    if (threadIdx.x == 0 && threadIdx.y == 0 && block_count > 0)
        atomicAdd(&burning[k], (unsigned int)block_count);   // integer add: order does not matter
}

// hits[c] = number of realisations in which cell c burned.
__global__ void hits_kernel(size_t n, int N, const uint8_t* __restrict__ state,
                            uint32_t* __restrict__ hits) {
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    uint32_t h = 0;
    for (int k = 0; k < N; ++k) {
        uint8_t s = state[(size_t)k * n + c];
        h += (s == BURNED || s == BURNING);
    }
    hits[c] = h;
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

    int dev = 0; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // ---------------------------------------------------------------- host setup
    Landscape land = make_landscape(rows, cols, seed);
    LandView L = land.view();
    Scenario S = {8.f, 3.f, 0.785f, 0.5f, 0.08f, 0.03f};
    const size_t n = (size_t)rows * cols;
    const size_t total = (size_t)N * n;

    std::vector<float> ps = make_slope_table(L);
    std::vector<StepParams> sp(N);
    std::vector<uint8_t> active(N), state(total);
    std::vector<uint16_t> res(total, 0);
    std::vector<uint32_t> steps(N, 0);

    for (int k = 0; k < N; ++k) {
        uint8_t* s = state.data() + (size_t)k * n;
        for (size_t c = 0; c < n; ++c) s[c] = L.fuel[c] ? UNBURNED : UNBURNABLE;
        RealParams rp = sample_realisation(L, S, seed, (uint32_t)k);
        sp[k] = make_step_params(rp);
        active[k] = rp.ig_i >= 0;                        // no burnable cell: nothing to simulate
        if (active[k]) {
            size_t ig = (size_t)rp.ig_i * cols + rp.ig_j;
            s[ig] = BURNING;
            res[(size_t)k * n + ig] = fuel_model(L.fuel[ig]).residence;
        }
    }

    // ---------------------------------------------------------------- device buffers
    uint8_t *d_fuel, *d_cur, *d_nxt, *d_active; float* d_ps; StepParams* d_sp;
    uint16_t* d_res; unsigned int* d_burning; uint32_t* d_hits;
    CUDA_CHECK(cudaMalloc(&d_fuel, n));
    CUDA_CHECK(cudaMalloc(&d_ps, 8 * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sp, N * sizeof(StepParams)));
    CUDA_CHECK(cudaMalloc(&d_active, N));
    CUDA_CHECK(cudaMalloc(&d_cur, total));
    CUDA_CHECK(cudaMalloc(&d_nxt, total));
    CUDA_CHECK(cudaMalloc(&d_res, total * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_burning, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_hits, n * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_fuel, L.fuel, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ps, ps.data(), 8 * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sp, sp.data(), N * sizeof(StepParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, active.data(), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cur, state.data(), total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, res.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // ---------------------------------------------------------------- time loop
    dim3 block(BX, BY);
    dim3 grid((cols + BX - 1) / BX, (rows + BY - 1) / BY, N);
    std::vector<unsigned int> burning(N);
    auto t0 = std::chrono::steady_clock::now();

    uint32_t t = 0;
    bool any_active = std::any_of(active.begin(), active.end(), [](uint8_t a) { return a != 0; });
    while (any_active && t < tmax) {
        CUDA_CHECK(cudaMemset(d_burning, 0, N * sizeof(unsigned int)));
        step_kernel<<<grid, block>>>(rows, cols, d_fuel, d_ps, d_sp, d_active, seed, t,
                                     d_cur, d_nxt, d_res, d_burning);
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_cur, d_nxt);
        ++t;
        CUDA_CHECK(cudaMemcpy(burning.data(), d_burning, N * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        // Same stopping rule as the serial loop, per realisation.
        bool changed = false;
        any_active = false;
        for (int k = 0; k < N; ++k) {
            if (!active[k]) continue;
            steps[k] = t;
            if (burning[k] == 0) { active[k] = 0; changed = true; }
            else any_active = true;
        }
        if (changed) CUDA_CHECK(cudaMemcpy(d_active, active.data(), N, cudaMemcpyHostToDevice));
    }

    const int threads = 256;
    hits_kernel<<<(unsigned int)((n + threads - 1) / threads), threads>>>(n, N, d_cur, d_hits);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> hits(n);
    CUDA_CHECK(cudaMemcpy(hits.data(), d_hits, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---------------------------------------------------------------- report (same as serial)
    uint64_t total_steps = 0;
    for (int k = 0; k < N; ++k) total_steps += steps[k];
    double bp_sum = 0; uint32_t bp_max = 0;
    for (size_t c = 0; c < n; ++c) { bp_sum += hits[c]; bp_max = std::max(bp_max, hits[c]); }
    printf("grid %dx%d  N=%d  tmax=%u  seed=%llu  device=%s\n", rows, cols, N, tmax,
           (unsigned long long)seed, prop.name);
    printf("time %.3f s   steps %llu   MCUPS %.1f\n", sec, (unsigned long long)total_steps,
           (double)n * total_steps / (sec * 1e6));
    printf("mean BP %.4f   max BP %.3f\n", bp_sum / ((double)n * N), (double)bp_max / N);
    printf("HASH %016llx\n", (unsigned long long)fnv1a(hits.data(), n * sizeof(uint32_t)));

    cudaFree(d_fuel); cudaFree(d_ps); cudaFree(d_sp); cudaFree(d_active); cudaFree(d_cur);
    cudaFree(d_nxt); cudaFree(d_res); cudaFree(d_burning); cudaFree(d_hits);
    return 0;
}
