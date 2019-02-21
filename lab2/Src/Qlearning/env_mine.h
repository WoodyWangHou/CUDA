#pragma once
/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

class env_mineCls
{
public:
	~env_mineCls();
	void init(int boardsize, int *board, int num_agent);
	void reset(int sid);
	void step(int sid, short* actions);
	void render(int* board, int sid);
	void clearboard(int sid);
public:
	int2 *d_state[2];
	float *d_reward;
	int2 *m_state;
private:
	int *d_board;
	int m_board_stride;
	int m_board_width;
	int m_board_height;
	int m_num_agent;
};