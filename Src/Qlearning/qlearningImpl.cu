/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdlib.h>
#include "draw_env.h"
#include "agent.h"
#include "common_def.h"

const int TABLE_SIZE = 4;
const int NUM_ACTIONS = 4;

// define number of agents 
const int NUM_AGENTS = 1;
const Agent* agents;

// Implementation of Agent
void Agent::qtable_init(int table_size, int num_actions) {
	this->m_h_qtable = new int**[table_size];
	for (int i = 0; i < table_size; ++i) {
		this->m_h_qtable[i] = new int*[table_size];
		for (int j = 0; j < num_actions; ++j) {
			this->m_h_qtable[i][j] = new int[num_actions];
			memset(this->m_h_qtable[i][j], 0, num_actions);
		}
	}
}

short Agent::getMaxQActionVal(int2* state) {
	int x = state->x;
	int y = state->y;

	short cand = RIGHT;
	for (int i = RIGHT; i <= TOP; ++i) {
		if (y == table_size - 1 && i == BOTTOM) continue;
		if (y == 0 && i == TOP) continue;
		if (x == table_size - 1 && i == RIGHT) {
			++cand;
			continue;
		}
		if (x == 0 && i == LEFT) continue;

		cand = this->m_h_qtable[y][x][cand] >= this->m_h_qtable[y][x][i] ? cand : i;
	}

	return cand;
}

// Implemetation of required functions
void agent_init() {
	if (!agents) {
		agents = new Agent;
	}
}

void agent_clearaction() {
	agents->resetAction();
}

float agent_adjustepsilon() {
	agents.m_epsilon -= 0.01f;
	return agents.m_epsilon;
}

short* agent_action(int2* cstate) {
	float seed = (float)(rand() % 100) / 100.0f;
	short ans = RIGHT;

	if (seed < agents->m_epsilon) {
		ans = rand() % 4;
	}
	else {
		short ans = agents->getMaxQActionVal(cstate);
	}

	agents->m_action = ans;

	return &ans;
}

void agent_update(int2* cstate, int2* nstate, float *rewards) {
	float alpha = 0.5;
	float delta = agents->getMaxQActionVal(nstate);
	float curVal = agents->m_h_qtable[cstate->y][cstate->x][agents->m_action];
	agents->m_h_qtable[cstate->y][cstate->x][agents->m_action] = curVal + alpha * (*(rewards) + agents->learningRate * delta - curVal);
}