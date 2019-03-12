/**
*	Author: Hou Wang
*   PID: A53241783
*	This file defines the data structure for grid
*/

#pragma once
namespace grid {
	const int NUMBER_OF_BITS = 4;
	const int NUMBER_OF_PARTICLES = 32;
	const int NUMBER_OF_KEYS = 16;

	class Grid {
#include <string>
	private:
		// both 32 long
		// for radix sort
		int *keys;
		int *value;

		// for particle counts
		int *count;
		int* cellId;
	public:
		void init();
		void sortKeys();
		void countParticles();
		void verify();
		std::string toString();
		~Grid() {
			if (count) {
				delete count;
				count = nullptr;
			}

			if (cellId) {
				delete cellId;
				cellId = nullptr;
			}

			if (keys) {
				delete keys;
				keys = nullptr;
			}

			if (value) {
				delete value;
				value = nullptr;
			}
		}
	};
}

#include <curand_kernel.h>
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