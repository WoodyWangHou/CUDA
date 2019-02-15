/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "agent.h"
#include "qlearningImpl.h"
#include "common_def.h"

#define DIMENSION 4

// Implemetation of required functions
void agent_init() {
	// update global variable
	initGlobalVariables();

	// init qtable, agent action states
	int actionMemSize = NUM_AGENT * sizeof(int);
	int qtableMemSize = DIMENSION * DIMENSION * NUM_ACTIONS * sizeof(float);

	CHECK(cudaMalloc((void **)&d_action, actionMemSize));
	CHECK(cudaMalloc((void **)&d_qtable, qtableMemSize));

	// to be updated for multi-agent
	dim3 block(1,1,1);
	dim3 grid(1,1,1);

	agentsInit <<<grid, block>>> (d_action, NUM_AGENT);
	qtableInit <<<grid, block>>> (d_qtable, DIMENSION * DIMENSION * NUM_ACTIONS);

	cudaDeviceSynchronize();
}

void agent_clearaction() {
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	agentsInit <<<grid, block>>> (d_action, NUM_AGENT);
}

float agent_adjustepsilon() {
	return decEpsilon();
}

// the pointer is pointing to memory in GPU
// need to return pointer to memory in GPU
short* agent_action(int2* cstate) {
	return &ans;
}

void agent_update(int2* cstate, int2* nstate, float *rewards) {

}