/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#define NUM_AGENT 128 //(256)
#define NUM_ACTIONS 4
#define LOG2_AGENT_STRIDE 7 // threadblock size (32*8)
#define ALIVE_CODE16 0x8000
#define ALIVE_MASK16 0x7fff


