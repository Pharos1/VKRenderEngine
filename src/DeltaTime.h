#ifndef DELTATIME
#define DELTATIME

#include <chrono>

namespace DT {
	float time;
	std::chrono::high_resolution_clock::time_point lastTime;

	static void update() {
		auto now = std::chrono::high_resolution_clock::now();
		time = std::chrono::duration<float, std::chrono::seconds::period>(now - lastTime).count();

		lastTime = std::chrono::high_resolution_clock::now();
	}
}

#endif