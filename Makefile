# -ffp-contract=off: stops the compiler fusing a*b+c differently in different loops,
# which would change results between serial and OpenMP builds.
CXXFLAGS = -O3 -std=c++17 -march=native -ffp-contract=off -Wall -Wextra
all: serial omp
serial: main_serial.cpp fire.h
	$(CXX) $(CXXFLAGS) -o $@ main_serial.cpp
omp: main_omp.cpp fire.h
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ main_omp.cpp
clean:
	rm -f serial omp
