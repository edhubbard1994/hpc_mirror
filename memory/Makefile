COMMON	= ../common

CXX	?= icpc
CXXFLAGS= -O3 -g -I. -std=c++11 -I$(COMMON)
#CXXFLAGS += -debug inline-debug-info
#CXXFLAGS += -no-vec
LDFLAGS	= -lrt

OMP = -fopenmp

CXXFLAGS += -DL1SIZE=`getconf LEVEL1_DCACHE_SIZE`
CXXFLAGS += -DL2SIZE=`getconf LEVEL2_CACHE_SIZE`
CXXFLAGS += -DL3SIZE=`getconf LEVEL3_CACHE_SIZE`

F90	?= ifort
F90FLAGS = -O3 -g

ifneq ($(TACC_MKL_LIB),)
#LDFLAGS	+= -lmkl_rt -L$(TACC_MKL_LIB)
LDFLAGS	+= -mkl=sequential
else
 ifneq ($(MKLROOT),)
  #LDFLAGS += -lmkl_core -lmkl_mc -lmkl_rt -L $(MKLROOT)/lib/intel64
  LDFLAGS +=  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
 else
  #LDFLAGS	+= -lblas -llapack -L/usr/lib64
  LDFLAGS	+= -l:liblapack.so.3 -l:libblas.so.3 -L/usr/lib64
 endif
endif

ifeq ($(CXX),icpc)
#CXXFLAGS += -xMIC-AVX512 #-no-vec
CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high #-no-vec
CXXFLAGS += -qopt-report=5
CXXFLAGS += -qopenmp-simd
F90FLAGS += -xCORE-AVX512 -qopt-zmm-usage=high -no-vec
F90FLAGS += -qopt-report=5
F90FLAGS += -qopenmp-simd
LDFLAGS += -lifcore
endif

ifeq ($(CXX),g++)
CXXFLAGS += -mtune=native
endif

ifneq ($(restrict),)
ifneq ($(restrict),0)
CXXFLAGS += -D__RESTRICT=restrict -restrict
endif
endif

ifneq ($(align),)
ifneq ($(align),0)
CXXFLAGS += -D__ALIGNMENT=$(align)
endif
endif

ifneq ($(unroll),)
ifneq ($(unroll),0)
CXXFLAGS += -DUNROLL=$(unroll)
endif
endif

ifneq ($(blocksize),)
ifneq ($(blocksize),0)
CXXFLAGS += -DBLOCKSIZE=$(blocksize)
endif
endif

ifneq ($(debug),)
ifneq ($(debug),0)
CXXFLAGS += -DDEBUG=1
endif
endif

ifneq ($(papi),)
ifneq ($(papi),0)
CXXFLAGS += -DWITH_PAPI -I$(TACC_PAPI_DIR)/include
LDFLAGS += -L$(TACC_PAPI_DIR)/lib -lpapi
endif
endif

EXEC = cache matmul ludec
EXEC_OMP = matmul_omp ludec_omp

all: $(EXEC)

%.o : %.f90
	$(F90) $(F90FLAGS) -c $<

%_omp.o : %.f90
	$(F90) $(OMP) $(F90FLAGS) -c $< -o $@

%_omp.o : %.cpp
	$(CXX) $(OMP) $(CXXFLAGS) -c $< -o $@

matmul: matmul.o matmul_kernels.o matmul_blas.o my_timer.o
	$(CXX) $(CXXFLAGS) -o matmul $^ $(LDFLAGS)

matmul_omp: matmul_omp.o matmul_kernels_omp.o matmul_blas.o my_timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

cache: cache.cpp my_timer.o
	$(CXX) $(CXXFLAGS) -o cache cache.cpp my_timer.o $(LDFLAGS)

ludec: ludec.cpp my_timer.o
	$(CXX) $(CXXFLAGS) -o ludec ludec.cpp my_timer.o $(LDFLAGS)

ludec_omp: ludec.cpp my_timer.o
	$(CXX) $(CXXFLAGS) $(OMP) -o ludec_omp ludec.cpp my_timer.o $(LDFLAGS)

my_timer.o: $(COMMON)/my_timer.h
	$(CXX) $(CXXFLAGS) -c $(COMMON)/my_timer.cpp -o my_timer.o

omp: $(EXEC_OMP)

clean:
	/bin/rm -fv $(EXEC) *.o *.optrpt
