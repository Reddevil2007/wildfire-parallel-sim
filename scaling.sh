#!/usr/bin/env bash
# Scaling study: median of 3 runs per thread count. Output is CSV you can paste into Excel.
# usage: bash scaling.sh [grid] [N] [tmax]     defaults: 512 16 2000  (takes a while)
cd "$(dirname "$0")" || exit 1
ROWS=${1:-512}; N=${2:-16}; TMAX=${3:-2000}; SEED=42
MAXT=$(nproc); THREADS="1"; t=2; while [ $t -le $MAXT ]; do THREADS="$THREADS $t"; t=$((t*2)); done
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_SCHEDULE=${OMP_SCHEDULE:-static}
echo "mode,threads,seconds,mcups"
for mode in stencil ensemble; do
  for t in $THREADS; do
    for r in 1 2 3; do
      OMP_NUM_THREADS=$t ./omp $ROWS $ROWS $N $TMAX $SEED $mode | awk -v m=$mode -v t=$t '/^time/{print m","t","$2","$7}'
    done | sort -t, -k3 -n | sed -n 2p
  done
done
