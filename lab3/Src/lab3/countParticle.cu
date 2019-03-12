/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <iostream>
#include <string>
#include "grid.h"

const int SUCCESS = 0;

// main function
int main(int argc, char* argv) {
	using namespace grid;
	Grid grid;

	// init particles and keys
	grid.init();

	// print particles and keys before sorting and counting
	std::cout << grid.toString() << std::endl;

	// sort keys
	grid.sortKeys();

	// count particles
	grid.countParticles();

	// print after sorting and counting
	std::cout << grid.toString() << std::endl;

	// verify the final result
	grid.verify();

	system("pause");
	return SUCCESS;
}

