#ifndef DELTATIME
#define DELTATIME

#include <GLFW/glfw3.h>

namespace DT {
	float time;
	float lastTime;

	static void update() {
		time = glfwGetTime() - lastTime;
		lastTime = glfwGetTime();
	}
}

#endif