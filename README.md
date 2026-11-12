# Parallel Wildfire Spread Simulation

A stochastic cellular-automaton (CA) wildfire spread model with Monte Carlo ensembles, used to estimate the
burn probability of every cell on a raster landscape. This repository has a **serial reference**, an
**OpenMP** implementation and a **CUDA** implementation. It was built as the course project for Parallel Computer Architecture and
Programming (PCAP) at Manipal Institute of Technology.

## How it works

- Each cell is `UNBURNABLE`, `UNBURNED`, `BURNING` or `BURNED`. On every step, an unburned cell may ignite from
  any of its 8 burning neighbours. The ignition probability depends on fuel type, wind speed and direction,
  slope and fuel moisture (spread terms after Alexandridis et al., 2008).
- Each Monte Carlo realisation draws its own wind, moisture and ignition point. The fraction of realisations
  in which a cell burns is its burn probability.
- The update is written in **gather form**: a cell reads its neighbours and writes only itself, so the
  parallel loops need no locks or atomics.
- Random numbers come from a **counter-based generator** keyed on (seed, realisation, cell, step). The output
  is therefore bit-identical whatever the number of threads, and every run prints a hash to prove it.

## Two levels of parallelism (OpenMP)

| Mode | What runs in parallel | Schedule |
|------|-----------------------|----------|
| `stencil` | the cell sweep inside one realisation (fine-grained) | static |
| `ensemble` | whole realisations, one buffer set per thread (coarse-grained) | dynamic, 1 |

## CUDA version

`main_cuda.cu` runs both levels on the GPU at once: one thread per cell, and one grid slice per realisation
(`blockIdx.z`). All realisations advance together, one kernel launch per time step. Each block counts its
burning cells with `__syncthreads_count` and adds them with a single `atomicAdd`, and a realisation that has
burned out is only copied forward.

GPU `expf`/`atanf` are accurate but not bit-identical to the CPU's, which could flip a rare ignition test. So
the wind, slope and moisture factors are computed once on the host (`fire_tables.h`), and the GPU only
multiplies and compares, built with `-fmad=false`. `check_tables.cpp` runs this table-driven kernel on the CPU
and reproduces the serial hash, so the GPU path computes exactly the same model.

## Build and run

Needs `g++` with OpenMP (Linux or WSL).

```bash
make                 # builds ./serial and ./omp
bash run_all.sh      # checks every OpenMP run reproduces the serial hash
bash scaling.sh      # CSV of time and MCUPS vs thread count (median of 3)
```

Run one binary directly:

```bash
./serial 512 512 16 2000 42
OMP_NUM_THREADS=8 ./omp 512 512 16 2000 42 ensemble
```

Arguments: `rows cols realisations max_steps seed [stencil|ensemble]`.

CUDA (needs `nvcc` and an NVIDIA GPU; Google Colab's free GPU works):

```bash
bash run_cuda.sh     # checks the table kernel on the CPU, then builds ./cuda and checks it on the GPU
./cuda 512 512 16 2000 42
```

## Results

Measured on a synthetic 512 × 512 landscape with 16 realisations:

| Threads | `stencil` speedup | `ensemble` speedup |
|---------|-------------------|--------------------|
| 16 | 5.0× | 5.8× |

Every OpenMP run reproduces the serial output hash.

One lesson learned: libgomp's default `schedule(runtime)` is `dynamic,1`, which made the two-thread cell
sweep take 206.9 s instead of 21.2 s. `main_omp.cpp` now defaults to static scheduling.

## Files

| File | Purpose |
|------|---------|
| `fire.h` | Shared kernel: cell states, fuel models, RNG, cell update, synthetic landscape |
| `main_serial.cpp` | Serial reference implementation |
| `main_omp.cpp` | OpenMP version (`stencil` and `ensemble` modes) |
| `fire_tables.h` | Table-driven cell update used by the CUDA version |
| `main_cuda.cu` | CUDA version (cells and realisations in parallel on the GPU) |
| `check_tables.cpp` | Runs the CUDA version's kernel on the CPU and checks it against serial |
| `run_all.sh` | Builds and checks OpenMP output against serial |
| `run_cuda.sh` | Builds and checks the CUDA version against serial |
| `scaling.sh` | Thread-scaling study |

## Limitations

The spread model is a simplified surrogate that has not been calibrated, the landscape is synthetic, and
results have not been validated against a real fire. GPU timings are not reported yet.

## Team

- Hrishit Tiwari
- Giridhar Sreekumar
- Riyan Dutta

Mentor: Dr. Neelima Bayyapu, School of Computer Science and Engineering, Manipal Institute of Technology, MAHE.
