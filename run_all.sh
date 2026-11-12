#!/usr/bin/env bash
# Build, then check that every OpenMP run reproduces the serial result (same HASH).
# usage: bash run_all.sh [grid] [N] [tmax]     defaults: 256 8 500
cd "$(dirname "$0")" || exit 1
sed -i 's/\r$//' Makefile run_all.sh 2>/dev/null   # strip Windows line endings
ROWS=${1:-256}; N=${2:-8}; TMAX=${3:-500}; SEED=42
make -s || { echo "BUILD FAILED"; exit 1; }
echo "cores available: $(nproc)"
echo "--- serial"
./serial $ROWS $ROWS $N $TMAX $SEED | tee /tmp/serial.out
REF=$(grep HASH /tmp/serial.out | awk '{print $2}')
FAIL=0
for mode in stencil ensemble; do
  for t in 1 2 4; do
    echo "--- omp $mode threads=$t"
    OMP_NUM_THREADS=$t ./omp $ROWS $ROWS $N $TMAX $SEED $mode | tee /tmp/omp.out | grep -E "time|HASH"
    H=$(grep HASH /tmp/omp.out | awk '{print $2}')
    if [ "$H" != "$REF" ]; then echo "MISMATCH vs serial"; FAIL=1; fi
  done
done
[ $FAIL -eq 0 ] && echo "PASS: all OpenMP runs match serial ($REF)" || echo "FAIL: hash mismatch"
