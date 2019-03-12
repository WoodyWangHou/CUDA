/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string>
#include <cstdlib>
#include "grid.h"
namespace grid {

	__global__ void pRadixSort(int *d_keys, int *d_value, int numOfRuns) {
		__shared__ int s_keys_values[NUMBER_OF_PARTICLES][2]; // store runs of sort
		int idx = threadIdx.x & 31;
		// load value into shared mem, coalesced
		s_keys_values[idx][1] = d_keys[idx];
		s_keys_values[idx][0] = d_value[idx];

		// radix sort
		unsigned int mask = 0x01;
		#pragma unroll
		for (int i = 0; i < numOfRuns; ++i) {
			int curKey = s_keys_values[idx][1];
			int curVal = s_keys_values[idx][0];
			unsigned int one = (curKey & mask) >> i;
			unsigned int zero = (~one) & 0x01;
			unsigned int prefixOne = one;
			unsigned int prefixZero = zero;

			// get prefix sums for one
			#pragma unroll
			for (int lane = 0; lane < 32; lane *= 2) {
				unsigned int tmp = __shfl_up(prefixOne, lane);
				if (idx >= lane) {
					prefixOne += tmp;
				}
			}

			// get prefix sums for zero
			#pragma unroll
			for (int lane = 0; lane < 32; lane *= 2) {
				unsigned int tmp = __shfl_up(prefixZero, lane);
				if (idx >= lane) {
					prefixZero += tmp;
				}
			}

			if (zero > 0) {
				// current bit is 0
				s_keys_values[prefixZero][1] = curKey;
				s_keys_values[prefixZero][0] = curVal;
			} else {
				// current bit is 1
				int laneId = blockDim.x % 32;
				int totalZero = __shfl(prefixZero, laneId, blockDim.x % 32);
				s_keys_values[prefixOne + totalZero][1] = curKey;
				s_keys_values[prefixOne + totalZero][0] = curVal;
			}

			mask <<= 1;
		}

		// copy from smem to gmem
		d_keys[idx] = s_keys_values[idx][1];
		d_value[idx] = s_keys_values[idx][0];
	}

	__global__ void pCountParticles(int *d_keys, int *d_value, int *d_count) {
		
	}

	void Grid::init() {
		this->count = new int[NUMBER_OF_KEYS];
		this->cellId = new int[NUMBER_OF_KEYS];
		this->keys = new int[NUMBER_OF_PARTICLES];
		this->value = new int[NUMBER_OF_PARTICLES];

		for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
			this->keys[i] = std::rand() % NUMBER_OF_KEYS;
			this->value[i] = i;
		}
	}
	
	void Grid::sortKeys() {
		dim3 block(NUMBER_OF_PARTICLES);
		int *d_keys;
		int *d_value;

		CHECK(cudaMalloc((void **)&d_keys, sizeof(int) * NUMBER_OF_PARTICLES));
		CHECK(cudaMalloc((void **)&d_value, sizeof(int) * NUMBER_OF_PARTICLES));
		CHECK(cudaMemcpy(d_keys, this->keys, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_value, this->value, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyHostToDevice));
		pRadixSort <<<1, block>>> (d_keys, d_value, NUMBER_OF_BITS);
		CHECK(cudaMemcpy(this->keys, d_keys, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(this->value, d_value, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyDeviceToHost));
	}

	void Grid::countParticles() {
		dim3 block(NUMBER_OF_PARTICLES);
		int *d_keys;
		int *d_value;
		int *d_count;
		int *d_cellId;
		
		CHECK(cudaMalloc((void **)&d_keys, sizeof(int) * NUMBER_OF_PARTICLES));
		CHECK(cudaMalloc((void **)&d_value, sizeof(int) * NUMBER_OF_PARTICLES));
		CHECK(cudaMalloc((void **)&d_count, sizeof(int) * NUMBER_OF_KEYS));
		CHECK(cudaMalloc((void **)&d_cellId, sizeof(int) * NUMBER_OF_KEYS));
		CHECK(cudaMemcpy(d_keys, this->keys, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_value, this->value, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyHostToDevice));

		pCountParticles << <1, block >> > (d_keys, d_value, d_count);

		CHECK(cudaMemcpy(this->keys, d_keys, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(this->value, d_value, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(this->count, d_count, sizeof(int) * NUMBER_OF_KEYS, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(this->cellId, d_cellId, sizeof(int) * NUMBER_OF_KEYS, cudaMemcpyDeviceToHost));
	}

	void Grid::verify() {
		for (int i = 0; i < NUMBER_OF_KEYS; i++) {
			if(i != *(this->cellId)) {// empty cell
				printf("%d %d \n", i, 0);
			}else{
				printf("%d %d" , *(this->cellId), ∗(this->count));
				this->cellId++;
				for (int k = 0; k < ∗(this->count); k++) {
					printf("%2d", *(this->value));
					this->value++;
				}
				printf("\n");
				this->count++;
			}
		}

	}

	std::string Grid::toString() {
		std::string str;
		str += "Particle Id: \t";
		for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
			std::string curVal = std::to_string(this->value[i]);
			str += curVal + (curVal.size() == 1 ? "  " : " ");
		}
		str += "\n";
		str += "Cell Id: \t";

		for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
			std::string curVal = std::to_string(this->keys[i]);
			str += curVal + (curVal.size() == 1 ? "  " : " ");
		}
		str += "\n";

		return str;
	}
}