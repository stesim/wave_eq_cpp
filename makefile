# The C++ compiler - this variable is used in the implicit rule for producing
# N.o from N.cc (or N.C)
CXX = g++

# The C compiler - this is used in two implicit rules:
# - the rule for producing N.o from N.c
# - the rule to link a single object file.
CC = gcc

NVCC = nvcc

# Preprocessor flags
CPPFLAGS = -I/usr/include/python3.3m -pthread
			# -DNDEBUG when the debugging code as given in section 8.3
            # has to be turned of

# We split up the compiler flags for convenience.

# Warning flags for C programs
WARNCFLAGS = -Wall # -W -Wshadow -Wpointer-arith -Wbad-function-cast \
	-Wcast-qual -Wcast-align -Wstrict-prototypes -Wmissing-prototypes \
	-Wmissing-declarations

# Warning flags for C++ programs
WARNCXXFLAGS = $(WARNCFLAGS) -Wold-style-cast -Woverloaded-virtual
# Debugging flags
DBGCXXFLAGS =  # -g
# Optimisation flags. Usually you should not optimise until you have finished
# debugging, except when you want to detect dead code.
OPTCXXFLAGS = -std=c++11 -O3 -march=native -DOPTI_MAX -DARMA_NO_DEBUG # -O2

# CXXFLAGS is used in the implicit rule for producing N.o from N.cc (or N.C)
CXXFLAGS = $(WARNCXXFLAGS) $(DBGCXXFLAGS) $(OPTCXXFLAGS)

# The linker flags and the libraries to link against.
# These are used in the implicit rule for linking using a single object file;
# we'll use them in our link rule too.
LDFLAGS = -L/opt/cuda/lib64 -larmadillo -lpython3.3m -lOpenCL -lcudart# -g
# Use Electric Fence to track down memory allocation problems.
LOADLIBES = # -lefence

#stesim
MYPROGRAM = wave_eq_cpp
# The program to build
PROGRAMS = $(MYPROGRAM)

# All the .cc files
SOURCES = $(wildcard *.cpp)
# And the corresponding .o files
OBJECTS = $(SOURCES:.cpp=.o)

NVSOURCES = CudaHelper.cu
NVFLAGS = -I/opt/cuda/include -lcudart -arch=sm_20

# The first target in the makefile is the default target. It's usually called
# "all".
all:	depend $(PROGRAMS)

# We assume we have one program (`ourprogram') to build, from all the object
# files derived from .cc files in the current directory.
$(MYPROGRAM):	$(OBJECTS)
	$(NVCC) -o nvobject.o $(NVFLAGS) -c $(NVSOURCES)
	$(CXX) $(LDFLAGS) -o $(MYPROGRAM) $(OBJECTS) nvobject.o $(LOADLIBES)

# "clean" removes files not needed after a build.
clean:
	rm -f $(OBJECTS) nvobject.o

# "realclean" removes everything that's been built.
realclean: clean
	rm -f $(PROGRAMS) .depend TAGS

# "force" forces a full rebuild.
force:
	$(MAKE) realclean
	$(MAKE)

# "depend" calculates the dependencies.
depend:
	rm -f .depend
	$(MAKE) .depend

# This is the actual dependency calculation.
.depend: $(SOURCES)
	rm -f .depend
# For each source, let the compiler run the preprocessor with the -M and -MM
# options, which causes it to output a dependency suitable for make.
	for source in $(SOURCES) ; do \
	  $(CXX) $(CXXFLAGS) -M -MM $$source | tee -a .depend ; \
	done

# Include the generated dependencies.
ifneq ($(wildcard .depend),'')
include .depend
endif

# Tell make that "all" etc. are phony targets, i.e. they should not be confused
# with files of the same names.
.PHONY: all clean realclean force depend
