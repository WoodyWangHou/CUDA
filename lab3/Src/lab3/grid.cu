/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cstdlib>

namespace grid {

	const int Grid::NUMBER_OF_PARTICLES = 32;
	const int NUMBER_OF_KEYS = 16;

	void Grid::init() {
		count = new int;
		keys = new int[NUMBER_OF_PARTICLES];
		value = new int[NUMBER_OF_PARTICLES];

		for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
			keys[i] = std::rand() % NUMBER_OF_KEYS;
			value[i] = i;
		}
	}
	
	void Grid::sortKeys() {
		dim3 block(NUMBER_OF_KEYS);
		int *d_keys;

		CHECK(cudaMalloc((void **)&d_keys, sizeof(int) * NUMBER_OF_KEYS));
		CHECK(cudaMemcpy(d_keys, cellId));
		CHECK(pRadixSort <<<1, block >>> (d_keys);

	}

	void Grid::countParticles() {
	}

	void Grid::verify() {
	}

	const char* Grid::toString() {
	}
}