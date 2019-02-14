/**
*	Author: Hou Wang
*   PID: A53241783
*	This file defines the data structure for agent
*/

#pragma once

extern const int TABLE_SIZE;
extern const int NUM_ACTIONS;

enum Action {
	RIGHT = 0,
	BOTTOM = 1,
	LEFT = 2,
	TOP = 3
};

struct Agent
{
	~Agent() {
		if (m_h_qtable) {
			for (int i = 0; i < table_size; ++i) {
				if (m_h_qtable[i]) {
					for (int j = 0; j < table_size; ++j) {
						if (m_h_qtable[i][j]) {
							delete m_h_qtable[i][j];
							m_h_qtable[i][j] = nullptr;
						}
					}

					delete m_h_qtable[i];
					m_h_qtable[i] = nullptr;
				}
			}

			delete m_h_qtable;
			m_h_qtable = nullptr;
		}
	}
	
	Agent(int s = TABLE_SIZE, int n = NUM_ACTIONS):table_size(s),num_actions(n) {
		this->learningRate = 0.1f;
		resetAction();
		epsilon_init();
		qtable_init(table_size, num_actions);
	}

	void qtable_init(int table_size = TABLE_SIZE, int num_actions = NUM_ACTIONS);
	void epsilon_init() {
		this->m_epsilon = 1.0f;
	}
	
	void resetAction() {
		this->m_action = -1;
	}
	short getMaxQActionVal(int2* state);

	float m_epsilon;
	float*** m_h_qtable;
	int m_action;
	int table_size;
	int num_actions;
	float learningRate;
};