make clean
CXX=icpc make nbody3
thread_num = (1 2 4 8 16 32 64 128)
for i in thread_num do
	export OMP_NUM_THREADS=i
	echo "number of threads: ${i}\n" 
	./nbody3 -n $1 -s $2 > /dev/null | grep "Average time" 
	echo "================" 

