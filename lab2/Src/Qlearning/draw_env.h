#pragma once
#include <chrono>
#include <iomanip> 
#include <sstream>


#define  BOARD_SIZE 46 //32
#define  MINE_COUNT (BOARD_SIZE*3)
#define  AGENT_CODE 0x2
#define  AGENT_MASK 0xfffffffd
#define  MINE 0x80000000
#define  FLAG  1
#define MINE_AGENT 0x80000002 //(AGENT_CODE | MINE);
#define FLAG_AGENT 0x00000003 // (AGENT_CODE | FLAG);

