/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Hou Wang
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <iostream>
#include "grid.h"

const int SUCCESS = 0;

void showGrid(Grid &grid) {
	std::cout << "Grid Cell Ids and Particles: " << std::endl;
	std::cout << grid.toString() << std::endl;
}

// main function
int main(int argc, char* argv) {
	using namespace grid::Grid;
	Grid grid;

	// init particles and keys
	grid.init();

	// print particles and keys before sorting and counting
	showGrid(grid);

	// sort keys
	grid.sortKeys();

	// count particles
	grid.countParticles();

	// print after sorting and counting
	showGrid(grid);

	// verify the final result
	grid.verify();

	return SUCCESS;
}

