/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include "grid.h"
namespace grid {

	__global__ void pRadixSort(int *d_keys, int *d_value, int numOfRuns) {
		__shared__ int s_keys_values[NUMBER_OF_PARTICLES][2]; // store runs of sort
		int idx = threadIdx.x;
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
			for (int lane = 1; lane <= 32; lane *= 2) {
				unsigned int tmp = __shfl_up(prefixOne, lane);
				unsigned int zeroTmp = __shfl_up(prefixZero, lane);
				if (idx >= lane) {
					prefixOne += tmp;
					prefixZero += zeroTmp;
				}
			}
			 
			int laneId = 31 % blockDim.x;
			unsigned int totalZero = __shfl(prefixZero, laneId, blockDim.x);

			if (zero > 0) {
				// current bit is 
				s_keys_values[prefixZero - 1][1] = curKey;
				s_keys_values[prefixZero - 1][0] = curVal;
			} else {
				// current bit is 1
				s_keys_values[prefixOne + totalZero - 1][1] = curKey;
				s_keys_values[prefixOne + totalZero - 1][0] = curVal;
			}

			mask <<= 1;
		}

		// copy from smem to gmem
		d_keys[idx] = s_keys_values[idx][1];
		d_value[idx] = s_keys_values[idx][0];
	}

	__global__ void pCountParticles(int *d_keys, int *d_cellId, int *d_count) {
		int idx = threadIdx.x;
		int laneId = idx & 0x1f;
		int val = d_keys[idx];
		int nval = __shfl_down(val, 1);
		if (idx == 31) nval = 0;

		int mask = __ballot(nval != val);
		int offset = __popc(mask & ((1 << laneId) - 1));
		int zcnt = __clz(mask & ((1 << laneId) - 1));
		int runcnt = zcnt - 31 + laneId;
		
		if (nval != val) {
			d_cellId[offset] = val;
			d_count[offset] = runcnt;
		}
	}

	void Grid::init() {
		this->count = (int *)malloc(sizeof(int) * NUMBER_OF_KEYS);
		this->cellId = (int *)malloc(sizeof(int) * NUMBER_OF_KEYS);
		memset(this->cellId, -1, sizeof(int) * NUMBER_OF_KEYS);
		this->keys = (int *)malloc(sizeof(int) * NUMBER_OF_PARTICLES);
		this->value = (int *)malloc(sizeof(int) * NUMBER_OF_PARTICLES);

		std::srand(time(NULL));

		for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
			this->keys[i] = std::rand() % NUMBER_OF_KEYS;
			this->value[i] = i;
		}
	}
	
	void Grid::sortKeys() {
		dim3 block(NUMBER_OF_PARTICLES);
		int *d_keys = NULL;
		int *d_value = NULL;

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
		int *d_count;
		int *d_cellId;
		
		CHECK(cudaMalloc((void **)&d_keys, sizeof(int) * NUMBER_OF_PARTICLES));
		CHECK(cudaMalloc((void **)&d_count, sizeof(int) * NUMBER_OF_KEYS));
		CHECK(cudaMalloc((void **)&d_cellId, sizeof(int) * NUMBER_OF_KEYS));
		CHECK(cudaMemcpy(d_keys, this->keys, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyHostToDevice));

		pCountParticles << <1, block >> > (d_keys, d_cellId, d_count);

		CHECK(cudaMemcpy(this->keys, d_keys, sizeof(int) * NUMBER_OF_PARTICLES, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(this->count, d_count, sizeof(int) * NUMBER_OF_KEYS, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(this->cellId, d_cellId, sizeof(int) * NUMBER_OF_KEYS, cudaMemcpyDeviceToHost));
	}

	void Grid::verify() {
		for (int i = 0; i < NUMBER_OF_KEYS; i++) {
			if(i != *(this->cellId)) {// empty cell
				printf("%d %d \n", i, 0);
			}else{
				printf("%d %d " , *(this->cellId), ∗(this->count));
				this->cellId++;
				for (int k = 0; k < ∗(this->count); k++) {
					printf("%2d ", *(this->value));
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