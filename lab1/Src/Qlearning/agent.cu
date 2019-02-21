/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "agent.h"
#include "common_def.h"

__device__ float *d_qtable;
__device__ short *d_action;

__device__ curandState *randState;

// epsilon
__device__ float epsilon;

// qLearning Paramters
__device__ float learningRate;		// discount factor
__device__ float gradientDec;		// learning Rate alpha

// Helpers:
__global__ void setup_kernel(curandState *state) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init((unsigned long long)(clock() + idx), idx, 0, state);
}

// Implementation:
__global__ void agentsInit(short *d_agentsActions, int size) {
	for (int i = 0; i < size; ++i) {
		d_agentsActions[i] = 0;
	}
}

__global__ void qtableInit(float *d_qtable, int size) {
	for (int i = 0; i < size; ++i) {
		d_qtable[i] = 0;
	}
}

__global__ void actionTaken(int2* cstate, short *d_action, float *d_qtable, curandState *state) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	int x = cstate[idx].x;
	int y = cstate[idx].y;
	short cand = RIGHT;
	// gama greedy strategy:
	curandState localState = *state;
	float seed = curand_uniform(&localState);

	if (seed < epsilon) {
		float actionSeed = curand_uniform(&localState) * 4;
		cand = (short)actionSeed;
	} else {
		for (short i = RIGHT; i <= TOP; ++i) {
			//if (x == 0 && i == LEFT) continue;
			//if (x == DIMENSION - 1 && i == RIGHT) continue;
			//if (y == 0 && i == TOP) continue;
			//if (y == DIMENSION - 1 && i == BOTTOM) continue;

			int tableIdx = i * (DIMENSION * DIMENSION) + y * DIMENSION + x;
			int candIdx = cand * (DIMENSION * DIMENSION) + y * DIMENSION + x;
			cand = d_qtable[tableIdx] > d_qtable[candIdx] ? i : cand;
		}
	}
	d_action[idx] = cand;
	*state = localState;
}

__global__ void qtableUpdate(int2* cstate, int2* nstate, float *rewards, short *d_action, float *d_qtable) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	short curAction = d_action[0];
	int cx = cstate[idx].x;
	int cy = cstate[idx].y;

	int nx = nstate[idx].x;
	int ny = nstate[idx].y;

	if (cx == nx && cy == ny && rewards[idx] != 0) {
		return; // not update qtable
	} else {
		
		// Find maximum next state expected value
		float max = 0.0f;
		for (short i = RIGHT; i <= TOP; ++i) {
			int tableIdx = i * (DIMENSION * DIMENSION) + ny * DIMENSION + nx;
			float val = d_qtable[tableIdx];
			max = val > max ? val : max;
		}

		// update formula
		int tableIdx = d_action[idx] * (DIMENSION * DIMENSION) + cy * DIMENSION + cx;
		float curQval = d_qtable[tableIdx];
		d_qtable[tableIdx] = curQval + gradientDec * (rewards[idx] + learningRate * max - curQval);
	}
	
}

__global__ void updateEpsilon() {
	epsilon -= 0.001f;
	
	if (epsilon < 0) {
		epsilon = 0;
	}
}

// Implementations for host API
void initAgents() {
	dim3 block(1,1,1);
	dim3 grid(1,1,1);
	int actionMemSize = NUM_AGENT * sizeof(int);

	CHECK(cudaMalloc((void **)&d_action, actionMemSize));
	CHECK(cudaMalloc((void **)&randState, sizeof(curandState)));

	agentsInit <<<grid, block>>> (d_action, NUM_AGENT);
	setup_kernel << <grid, block >> >(randState);
	cudaDeviceSynchronize();
}

void initQTable() {
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);

	// init qtable, agent action states
	int qtableMemSize = DIMENSION * DIMENSION * NUM_ACTIONS * sizeof(float);
	CHECK(cudaMalloc((void **)&d_qtable, qtableMemSize));
	qtableInit <<<grid, block >>> (d_qtable, DIMENSION * DIMENSION * NUM_ACTIONS);
	cudaDeviceSynchronize();
}

void initGlobalVariables() {
	float ep = 1.0;
	CHECK(cudaMemcpyToSymbol(epsilon, &ep, sizeof(float)));

	float lr = 0.5;
	float gd = 0.5;
	CHECK(cudaMemcpyToSymbol(learningRate, &lr, sizeof(float)));
	CHECK(cudaMemcpyToSymbol(gradientDec, &gd, sizeof(float)));
}

float decEpsilon() {
	updateEpsilon << <1, 1 >> >();
	float h_epsilon = 0.0f;
	CHECK(cudaMemcpyFromSymbol(&h_epsilon, epsilon, sizeof(float)));
	return h_epsilon;
}

void takeAction(int2* cstate) {
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	actionTaken <<<grid, block >>> (cstate, d_action, d_qtable, randState);
}

void updateAgents(int2* cstate, int2* nstate, float *rewards) {
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	qtableUpdate <<<grid, block >>> (cstate, nstate, rewards, d_action, d_qtable);
}