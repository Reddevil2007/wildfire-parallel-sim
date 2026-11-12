#!/usr/bin/env bash
# Build the CUDA version and check it against the serial result (same HASH).
#   1. check_tables: the GPU's table-driven kernel run on the CPU (no GPU needed)
#   2. cuda:         the real GPU run (needs nvcc and an NVIDIA GPU, e.g. Google Colab)
# usage: bash run_cuda.sh [grid] [N] [tmax]     defaults: 256 8 500
cd "$(dirname "$0")" || exit 1
sed -i 's/\r$//' Makefile run_cuda.sh 2>/dev/null   # strip Windows line endings
ROWS=${1:-256}; N=${2:-8}; TMAX=${3:-500}; SEED=42
CXXFLAGS="-O3 -std=c++17 -march=native -ffp-contract=off -Wall -Wextra"
# -fmad=false: no fused multiply-add on the GPU, so products round exactly as on the CPU.
NVCCFLAGS="-O3 -std=c++17 -arch=native -fmad=false -Xcompiler -ffp-contract=off"

make -s serial || { echo "BUILD FAILED (serial)"; exit 1; }
${CXX:-g++} $CXXFLAGS -o check_tables check_tables.cpp || { echo "BUILD FAILED (check_tables)"; exit 1; }

echo "--- serial"
REF=$(./serial $ROWS $ROWS $N $TMAX $SEED | tee /tmp/serial.out | grep HASH | awk '{print $2}')
cat /tmp/serial.out
FAIL=0

echo "--- table kernel on CPU"
H=$(./check_tables $ROWS $ROWS $N $TMAX $SEED | grep HASH | awk '{print $2}')
echo "HASH $H"
[ "$H" == "$REF" ] || { echo "MISMATCH vs serial"; FAIL=1; }

if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  nvcc $NVCCFLAGS -o cuda main_cuda.cu || { echo "BUILD FAILED (cuda)"; exit 1; }
  echo "--- cuda"
  ./cuda $ROWS $ROWS $N $TMAX $SEED | tee /tmp/cuda.out
  H=$(grep HASH /tmp/cuda.out | awk '{print $2}')
  [ "$H" == "$REF" ] || { echo "MISMATCH vs serial"; FAIL=1; }
else
  echo "--- cuda: skipped (needs nvcc and an NVIDIA GPU)"
fi

[ $FAIL -eq 0 ] && echo "PASS: all runs match serial ($REF)" || echo "FAIL: hash mismatch"
