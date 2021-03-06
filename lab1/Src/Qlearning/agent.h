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
		system("pause");													   \
        exit(1);                                                               \
    }                                                                          \
}

#define DIMENSION 4

enum Action {
	RIGHT = 0,
	BOTTOM = 1,
	LEFT = 2,
	TOP = 3
};

// global Q table
// logically it is a 3d matrix
// each cell (x,y,action) represents Q(s, action)
extern __device__ float* d_qtable;

// Each agent needs to keep track of its own:
// 1. action
// 2. if it is alive (represented using value -1)
// this data structure is structure of arrays
extern __device__ short *d_action;

// functions to init agents data structure
void initGlobalVariables();
float decEpsilon();
void initAgents();
void initQTable();
void takeAction(int2* cstate);
void updateAgents(int2* cstate, int2* nstate, float *rewards);