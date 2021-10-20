make clean
CXX=icpc make nbody3
export OMP_NUM_THREADS=$3
./nbody3 -n $1 -s $2
