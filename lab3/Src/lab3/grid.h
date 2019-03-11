/**
*	Author: Hou Wang
*   PID: A53241783
*	This file defines the data structure for grid
*/

namespace grid {
	const int NUMBER_OF_PARTICLES;
	const int NUMBER_OF_KEYS;
	class Grid {
	private:
		int *count;
		// both 32 long, keys represents the 
		int *keys;
		int *value;
	public:
		void init();
		void sortKeys();
		void countParticles();
		void verify();
		const char* toString();
		~Grid() {
			if (count) {
				delete count;
				count = nullptr;
			}

			if (cellId) {
				delete cellId;
				cellId = nullptr;
			}

			if (particleId) {
				delete particleId;
				particleId = nullptr;
			}
		}
	};
}

#pragma once
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