#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_LEFT_HANDED

#include <GLM/glm.hpp>
#include <GLM/mat4x4.hpp>
#include <GLM/vec4.hpp>
#include <GLM/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <vector>
#include <map>
#include <optional>
#include <set>
#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <fstream>
#include <filesystem>
#include <unordered_map>

#include "Application.h"
#include "Camera.h"
#include "DeltaTime.h"

namespace App = Application;

void draw(VkCommandBuffer commandBuffer, uint32_t imageIndex);
void processInput();
void processMouse(GLFWwindow* window, double xPos, double yPos);

Camera cam({ 0.f, 1.f, -3.f });

int main() {
	std::string f = std::filesystem::current_path().string(); //Get working directory.
	if (strcmp(f.substr(f.find_last_of("\\") + 1).c_str(), "VKRenderEngine") == 0) {
		system("cd Shaders && spv.bat");
	}
	std::filesystem::current_path(std::filesystem::path(__FILE__).parent_path().parent_path()); //Working dir = solution path

	App::initVulkan();
	glfwSetCursorPosCallback(App::window, processMouse);
	cam.updateView();

	App::model = glm::mat4(1.f);
	App::proj = glm::perspective(glm::radians(45.0f), App::swapChainExtent.width / (float)App::swapChainExtent.height, 0.1f, 100.0f);
	//cam.view = glm::translate(glm::mat4(1.f), glm::vec3(0, 0.f, 3.f));
	
	while (!glfwWindowShouldClose(App::window)) {
		glfwPollEvents();
		DT::update();
		processInput();
		App::view = cam.view;

		App::drawFrame(draw);
	}

	vkDeviceWaitIdle(App::device);
	App::cleanup();

	return 0;
}
void draw(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
	vkCmdDrawIndexed(commandBuffer, App::indices.size(), 1, 0, 0, 0);
}
void processInput() {
	cam.processInput(App::window);

	if (glfwGetKey(App::window, GLFW_KEY_E)) glfwSetWindowShouldClose(App::window, GLFW_TRUE);
}
void processMouse(GLFWwindow* window, double xPos, double yPos) {
	cam.processMouse(xPos, yPos);
}