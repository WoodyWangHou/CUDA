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
Agent *agents;

void agent_init() {
	// update global variable
	agents = new Agent;
	agents->init(NUM_AGENT);
}

void agent_clearaction() {
	agents->initAgents();
}

float agent_adjustepsilon() {
	return agents->decEpsilon();
}

// the pointer is pointing to memory in GPU
// need to return pointer to memory in GPU
short* agent_action(int2* cstate) {
	agents->takeAction(cstate);
	return agents->getActions();
}

void agent_update(int2* cstate, int2* nstate, float *rewards) {
	agents->updateAgents(cstate,nstate,rewards);
}