/**
*	Author: Hou Wang
*   PID: A53241783
*	This file defines the data structure for agent
*/

#pragma once
#include <curand_kernel.h>
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
		system("pause");													   \
        exit(1);                                                               \
    }                                                                          \
}

enum Action {
	RIGHT = 0,
	BOTTOM = 1,
	LEFT = 2,
	TOP = 3
};


class Agent {
private:
	// Note: the following private fields are by default __device__ (implicit)

	// global Q table
	// logically it is a 3d matrix
	// each cell (x,y,action) represents Q(s, action)
	// the mapping is:
	// |(x1,y1,a1)||(x1,y1,a2)|(x1,y1,a3)|(x1,y1,a4)|(x2,y2,a1)...
	float* d_qtable;

	// Each agent needs to keep track of its own:
	// 1. action
	// 2. if it is alive (represented using value -1)
	// this data structure is structure of arrays
	short *d_action;

	// alive status
	bool *d_isAlive;

	curandState *randState;
	// epsilon
	float *epsilon;

	// qLearning Paramters
	float learningRate;		// discount factor
	float gradientDec;		// learning Rate alpha

	// define grid and block for all functions to use
	int numOfAgents;
	dim3 block;
	dim3 grid;

	void initQTable();
	// functions to init agents data structure
	void initGlobalVariables();

public:
	~Agent() {
		cudaFree(randState);
		cudaFree(epsilon);
		cudaFree((void *)d_qtable);
	}
	void init();
	void initAgents();
	float decEpsilon();
	void takeAction(int2* cstate);
	void updateAgents(int2* cstate, int2* nstate, float *rewards);
	short* getActions() {
		return this->d_action;
	}
};