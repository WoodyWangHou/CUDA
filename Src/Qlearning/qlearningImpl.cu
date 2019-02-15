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

// Implemetation of required functions
void agent_init() {
	// update global variable
	initGlobalVariables();

	// init qtable, agent action states
	int actionMemSize = NUM_AGENT * sizeof(int);
	int qtableMemSize = DIMENSION * DIMENSION * NUM_ACTIONS * sizeof(float);

	CHECK(cudaMalloc((void **)&d_action, actionMemSize));
	CHECK(cudaMalloc((void **)&d_qtable, qtableMemSize));

	initAgents();
	initQTable();
}

void agent_clearaction() {
	initAgents();
}

float agent_adjustepsilon() {
	return decEpsilon();
}

// the pointer is pointing to memory in GPU
// need to return pointer to memory in GPU
short* agent_action(int2* cstate) {
	updateActions(cstate);
	return d_action;
}

void agent_update(int2* cstate, int2* nstate, float *rewards) {

}