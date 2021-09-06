COMMON	= ../common

OMPFLAGS ?= -fopenmp
DBG      ?=
CXX      ?= g++
CXXFLAGS  = -O3 -I. -I$(COMMON) $(DBG)
CXXFLAGS += -g

CXXFLAGS += -std=c++11
#CXXFLAGS += -DENABLE_PARALLEL_CXX
#CXXFLAGS += -std=c++17 -DENABLE_CXX17

ifeq ($(CXX),icpc)
CXXFLAGS += -xHost #-no-vec
CXXFLAGS += -qopt-report=5
CXXFLAGS += -Wno-unknown-pragmas # Disable warning about OpenMP pragma no defined.
#CXXFLAGS += -debug inline-debug-info
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

ifneq ($(dtype),)
ifeq ($(dtype),float)
CXXFLAGS += -DUSE_FLOAT
endif
endif

ifneq ($(func),)
ifneq ($(func),0)
CXXFLAGS += -DACC_FUNC=$(func)
endif
endif

EXEC = sort

all: $(EXEC)

# Load common make options
include $(COMMON)/Makefile.common
LDFLAGS	  = $(COMMON_LIBS)

-include *.d

sort: sort.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sort_omp: sort_omp.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

clean: clean_common
	/bin/rm -fv $(EXEC) *.o *.optrpt *.d
