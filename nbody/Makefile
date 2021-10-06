COMMON	= ../common

DBG      ?= -g
CXX      ?= g++
CXXFLAGS  = -O3 -I. -std=c++11 -I$(COMMON) $(DBG)

#CXXFLAGS += -no-vec
CXXFLAGS += -Wfatal-errors

ifeq ($(CXX),icpc)
  CXXFLAGS += -xHost
# CXXFLAGS += -D__RESTRICT=restrict -restrict
# CXXFLAGS += -xavx2
# CXXFLAGS += -xMIC-AVX512
# CXXFLAGS += -xCORE-AVX512
#  CXXFLAGS += -qopt-zmm-usage=high
#  CXXFLAGS += -no-prec-sqrt
  CXXFLAGS += -qopt-report=5
  CXXFLAGS += -Wunknown-pragmas # Disable warning about OpenMP pragma no defined.
  CXXFLAGS += -qopenmp-simd
# CXXFLAGS += -prec-sqrt -prec-div
endif

ifneq ($(restrict),)
  do_restrict=0
  ifeq ($(restrict),yes)
    do_restrict=1
  endif
  ifneq ($(restrict),0)
    do_restrict=1
  endif

  ifeq ($(do_restrict),1)
    CXXFLAGS += -D__RESTRICT=restrict -restrict
  endif
endif

ifeq ($(CXX),g++)
  CXXFLAGS += -mtune=native
endif

ifneq ($(format),)
ifeq ($(format),aos)
  CXXFLAGS += -DEnable_ArrayOfStructures
endif
ifeq ($(format),soa)
  CXXFLAGS += -DEnable_StructureOfArrays
endif
endif

_USER_SET_ALIGNMENT=0
ifneq ($(align),)
ifneq ($(align),0)
  _USER_SET_ALIGNMENT=1
  CXXFLAGS += -D__ALIGNMENT=$(align)
endif
endif

ifeq ($(_USER_SET_ALIGNMENT),0)
  CXXFLAGS += -D__ALIGNMENT=`getconf LEVEL1_DCACHE_LINESIZE`
endif

ifneq ($(vcl),)
  ifeq ($(vcl),128)
    CXXFLAGS += -D__ENABLE_VCL_SIMD=128 -Ivcl
  else ifeq ($(vcl),256)
    CXXFLAGS += -D__ENABLE_VCL_SIMD=256 -Ivcl
  else ifeq ($(vcl),512)
    CXXFLAGS += -D__ENABLE_VCL_SIMD=512 -Ivcl
  else ifneq ($(vcl),0)
    CXXFLAGS += -D__ENABLE_VCL_SIMD=1 -Ivcl
  endif
endif

EXEC = nbody3

all: $(EXEC)

# Load common make options
include $(COMMON)/Makefile.common
LDFLAGS	  = $(COMMON_LIBS)

-include *.d

nbody3: nbody3.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o nbody3 $^ $(LDFLAGS)

simd: simd.o alias.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o simd $^ $(LDFLAGS)

clean: clean_common
	/bin/rm -fv $(EXEC) *.o *.optrpt *.d *.s
