/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "agent.h"
#include "common_def.h"

// Helpers:
__global__ void setup_kernel(curandState *state) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init((unsigned long long)(clock() + idx), idx, 0, &state[idx]);
}

__global__ void agentsInit(short *d_agentsActions, int size) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_agentsActions[idx] = 0;
}

__global__ void qtableInit(volatile float *d_qtable, int size) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_qtable[idx] = 0;
}

// Implementation:
__global__ void actionTaken(
	int2* cstate, 
	short *d_action, 
	volatile float *d_qtable, 
	curandState *state, 
	float *ep) {
	float epsilon = *ep;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	int x = cstate[idx].x;
	int y = cstate[idx].y;
	short cand = RIGHT;
	// gama greedy strategy:
	curandState localState = state[idx];
	float seed = curand_uniform(&localState);

	if (seed < epsilon) {
		float actionSeed = curand_uniform(&localState) * 4;
		cand = (short)actionSeed;
	} else {
		for (short i = RIGHT + 1; i <= TOP; ++i) {
			int tableIdx = i * (DIMENSION * DIMENSION) + y * DIMENSION + x;
			int candIdx = cand * (DIMENSION * DIMENSION) + y * DIMENSION + x;
			cand = d_qtable[tableIdx] > d_qtable[candIdx] ? i : cand;
		}
	}
	d_action[idx] = cand;
	state[idx] = localState;
}

__global__ void qtableUpdate(
	int2* cstate, 
	int2* nstate, 
	float *rewards, 
	short *d_action, 
	volatile float *d_qtable, 
	float gradientDec,
	float learningRate) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	short curAction = d_action[idx];
	int cx = cstate[idx].x;
	int cy = cstate[idx].y;

	int nx = nstate[idx].x;
	int ny = nstate[idx].y;

	if (cx == nx && cy == ny) return;

	// Find maximum next state expected value
	int maxIdx = ny * DIMENSION + nx;
	float max = d_qtable[maxIdx];
	for (short i = RIGHT + 1; i <= TOP; ++i) {
		int tableIdx = i * (DIMENSION * DIMENSION) + ny * DIMENSION + nx;
		float val = d_qtable[tableIdx];
		max = val > max ? val : max;
	}

	// update formula
	int tableIdx = d_action[idx] * (DIMENSION * DIMENSION) + cy * DIMENSION + cx;
	d_qtable[tableIdx] += gradientDec * (rewards[idx] + learningRate * max - d_qtable[tableIdx]);
}

__global__ void updateEpsilon(float *epsilon) {
	float val = *epsilon;
	(*epsilon) = val - 0.005f;
	
	if (val < 0) {
		(*epsilon) = 0;
	}
}

__global__ void epsilonInit(float *epsilon) {
	*epsilon = 1.0f;
}

// Implementations for host API
void Agent::init(int numOfAgents) {
	// update global variable
	block = dim3(numOfAgents);
	grid = dim3(1);
	this->numOfAgents = numOfAgents;
	this->initGlobalVariables();
	this->initAgents();
	this->initQTable();
}

void Agent::initAgents() {
	int actionMemSize = this->numOfAgents * sizeof(short);

	CHECK(cudaMalloc((void **)&d_action, actionMemSize));
	CHECK(cudaMalloc((void **)&randState, this->numOfAgents * sizeof(curandState)));

	agentsInit <<<grid, block>>> (d_action, this->numOfAgents);
	setup_kernel <<<grid, block >>>(randState);
	cudaDeviceSynchronize();
}

void Agent::initQTable() {
	// init qtable, agent action states
	int qtableMemSize = DIMENSION * DIMENSION * NUM_ACTIONS * sizeof(float);
	CHECK(cudaMalloc((void **)&d_qtable, qtableMemSize));
	qtableInit <<<grid, block >>> (d_qtable, DIMENSION * DIMENSION * NUM_ACTIONS);
	cudaDeviceSynchronize();
}

void Agent::initGlobalVariables() {
	float ep = 1.0;
	CHECK(cudaMalloc((void **)&epsilon, sizeof(float)));
	epsilonInit <<<1, 1 >>> (epsilon);

	this->learningRate = 0.20f;
	this->gradientDec = 0.25f;
}

float Agent::decEpsilon() {
	updateEpsilon <<<1, 1 >>>(this->epsilon);
	float h_epsilon = 0.0f;
	CHECK(cudaMemcpy(&h_epsilon, epsilon, sizeof(float), cudaMemcpyDeviceToHost));
	return h_epsilon;
}

void Agent::takeAction(int2* cstate) {
	actionTaken <<<grid, block >>> (cstate, d_action, d_qtable, randState, this->epsilon);
}

void Agent::updateAgents(int2* cstate, int2* nstate, float *rewards) {
	qtableUpdate <<<grid, block >>> (cstate, nstate, rewards, d_action, d_qtable, gradientDec, learningRate);
}