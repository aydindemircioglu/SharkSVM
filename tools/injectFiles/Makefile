CXX ?= g++
CFLAGS = -fopenmp -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: cmake

cmake: .FORCE

.FORCE: 
	rm -rf ./build; mkdir build; cd build; cmake ..; make all; cp ../bin/SharkSVM-predict ..; cp ../bin/SharkSVM-train ..;

	
