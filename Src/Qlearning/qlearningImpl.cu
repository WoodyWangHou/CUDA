/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include "draw_env.h"
#include "agent.h"
#include "common_def.h"

#define DIMENSION 4
#define NUM_ACTIONS 4
#define NUM_AGENTS 1

int actionMemSize = 0;
int qtableMemSize = 0;

// TODO: implement the following

__global__ void agentsInit(int *d_agentsActions) {
	d_agentsActions[0] = 0;
}
__global__ void qtableInit(float *d_qtable, int size) {
	for (int i = 0; i < size; ++i) {
		d_qtable[i] = 0;
	}
}

__global__ void resetAgents(int* d_agentsActions) {
}
__global__ void agentsUpdate(int2* cstate, int2* nstate, float *rewards) {
}

__global__ void updateEpsilon() {
	epsilon -= 0.1f;
}

// Implemetation of required functions
void agent_init() {

	// init host data
	actionMemSize = NUM_AGENTS * sizeof(int);
	h_action = (int *)malloc(actionMemSize);
	qtableMemSize = DIMENSION * DIMENSION * NUM_ACTIONS * sizeof(float);
	h_qtable = (float *)malloc(qtableMemSize);

	// allocate memory for cuda
	int* d_action;
	float* d_qtable;

	CHECK(cudaMalloc((void **)&d_action, actionMemSize));
	CHECK(cudaMalloc((void **)&d_qtable, qtableMemSize));
	CHECK(cudaMemcpy(d_action, h_action, actionMemSize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_qtable, h_qtable, qtableMemSize, cudaMemcpyHostToDevice));

	// to be updated for multi-agent
	dim3 block(1,1,1);
	dim3 grid(1,1,1);

	agentsInit <<<block, grid >>> (d_action);
	qtableInit <<<block, grid >>> (d_qtable, DIMENSION * DIMENSION * NUM_ACTIONS);

	// copy back
	CHECK(cudaMemcpy(h_action, d_action, actionMemSize, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_qtable, d_qtable, qtableMemSize, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	cudaFree(d_action);
	cudaFree(d_qtable);
}

void agent_clearaction() {
	
}

float agent_adjustepsilon() {
	updateEpsilon<<<1,1>>>();
	float *h_epsilon = (float *)malloc(sizeof(float));
	CHECK(cudaMemcpy(h_epsilon, &epsilon, sizeof(float), cudaMemcpyDeviceToHost));
	return *h_epsilon;
}

short* agent_action(int2* cstate) {
	short ans = 0;
	return &ans;
}

void agent_update(int2* cstate, int2* nstate, float *rewards) {
}