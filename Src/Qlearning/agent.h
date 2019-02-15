/**
*	Author: Hou Wang
*   PID: A53241783
*	This file defines the data structure for agent
*/

#pragma once

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

enum Action {
	DEAD = -1,
	RIGHT = 0,
	BOTTOM = 1,
	LEFT = 2,
	TOP = 3
};

// global Q table
// logically it is a 3d matrix
// each cell (x,y,action) represents Q(s, action)
float* h_qtable;

// Each agent needs to keep track of its own:
// 1. action
// 2. if it is alive (represented using value -1)
// this data structure is structure of arrays
int *h_action;

// epsilon
__device__ float epsilon = 1.0;

// qLearning Paramters
__device__ float learningRate = 0.2;		// discount factor
__device__ float gradientDec = 0.1;		// learning Rate alpha

// functions to init agents data structure
//__global__ void agentsInit(int* d_agentsActions);
//__global__ void qtableInit(float* d_qtable, int size);
