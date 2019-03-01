/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wa
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>

#pragma once

void agent_init();
void agent_clearaction();
float agent_adjustepsilon();
short* agent_action(int2* cstate);
void agent_update(int2* cstate, int2* nstate, float *rewards);