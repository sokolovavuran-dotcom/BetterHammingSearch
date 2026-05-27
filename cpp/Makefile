# Build all C++ benchmarks.
#
# Default compiler is g++ (MSYS2 MinGW on Windows, system g++ on Linux/Mac).
# Override with: make CXX=clang++
#
# Flags:
#   -O3            full optimization
#   -std=c++20     std::popcount, structured bindings
#   -march=native  use SIMD instructions available on this CPU (popcnt, AVX2)
#   -Wall -Wextra  warnings on
#
# Build:    make
# Clean:    make clean
# Run:      ./linear_scan --dataset siftsmall --radius 20 --query_count 10

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++20 -march=native -Wall -Wextra
LDFLAGS  ?=

BINS = linear_scan qadp_normal qadp_small mih

ifeq ($(OS),Windows_NT)
    EXE = .exe
else
    EXE =
endif

all: $(addsuffix $(EXE),$(BINS))

linear_scan$(EXE): linear_scan.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ linear_scan.cpp $(LDFLAGS)

qadp_normal$(EXE): qadp_normal.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ qadp_normal.cpp $(LDFLAGS)

qadp_small$(EXE): qadp_small.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ qadp_small.cpp $(LDFLAGS)

mih$(EXE): mih.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ mih.cpp $(LDFLAGS)

clean:
	rm -f $(addsuffix $(EXE),$(BINS))

.PHONY: all clean
