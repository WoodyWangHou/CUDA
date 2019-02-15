/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "agent.h"
#include "common_def.h"

__device__ float *d_qtable;
__device__ short   *d_action;

// epsilon
__device__ float epsilon;

// qLearning Paramters
__device__ float learningRate;		// discount factor
__device__ float gradientDec;		// learning Rate alpha

// TODO: implement the following

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

__global__ void actionsUpdate(int2* cstate, short *d_agentsActions, int size) {
}

__global__ void agentsUpdate(int2* cstate, int2* nstate, float *rewards) {

}

__global__ void updateEpsilon() {
	epsilon -= 0.1f;
}

void initAgents() {
	dim3 block(1,1,1);
	dim3 grid(1,1,1);

	agentsInit <<<grid, block>>> (d_action, NUM_AGENT);
	cudaDeviceSynchronize();
}

void initQTable() {
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	qtableInit << <grid, block >> > (d_qtable, DIMENSION * DIMENSION * NUM_ACTIONS);
	cudaDeviceSynchronize();
}

void initGlobalVariables() {
	float ep = 1.0;
	CHECK(cudaMemcpyToSymbol(epsilon, &ep, sizeof(float)));

	float lr = 0.1;
	float gd = 0.2;
	CHECK(cudaMemcpyToSymbol(learningRate, &lr, sizeof(float)));
	CHECK(cudaMemcpyToSymbol(gradientDec, &gd, sizeof(float)));
}

float decEpsilon() {
	updateEpsilon << <1, 1 >> >();
	float h_epsilon = 0.0f;
	CHECK(cudaMemcpyFromSymbol(&h_epsilon, epsilon, sizeof(float)));
	return h_epsilon;
}

void updateActions(int2* cstate) {
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	actionsUpdate <<<grid, block >>> (cstate, d_action, NUM_AGENT);
}