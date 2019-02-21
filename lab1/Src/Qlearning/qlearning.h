#pragma once


#include "env_mine.h"
#include "common_def.h"

class qlearningCls
{
public:
	~qlearningCls();
	void init(int boardsize, int *board);
	int learning(int *board, unsigned int &episode, unsigned int &steps);
	int alive_agent(int* board);
	void adjustpsilon() { m_epsilon = std::max(m_minepsilon, m_epsilon - m_epsilon_step); }
	int checkstatus(int* board, int2* state, unsigned int &fagent);
public:
	float m_epsilon;
private:
	env_mineCls env;
	//qlearning_agentCls agent;
	int m_sid;
	bool m_newepisode;
	unsigned int m_episode;
	unsigned int m_steps;
	int m_boardsize;
	
	float m_minepsilon;
	float m_epsilon_step;

};