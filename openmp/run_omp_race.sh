#!/bin/bash

n=${1:-100000}

echo "threads, critical, atomic, reduction-manual, reduction-auto, ${n}"
for nthreads in 1 2 4 8 16 32 64; do
   echo -n "$nthreads, "
   OMP_NUM_THREADS=$nthreads ./omp_race -n ${n} -i 10 | tail -n4 | awk '{printf("%f, ", $6)}END{printf("\n")}'
done
