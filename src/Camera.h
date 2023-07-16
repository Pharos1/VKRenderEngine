#ifndef CAMERA
#define CAMERA

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

#include <iostream>

#include "DeltaTime.h"

struct Camera {
	glm::mat4 view{ 1 };
	float pitch = 0.f;
	float yaw = 90.f;
	bool firstMouse = true;
	double lastX{}, lastY{};

	//Optional
	float camSpeed = 7.5f;
	float mouseSensitivity = 0.1f;
	glm::vec3 camPos{};
	glm::vec3 camFront = glm::vec3(0.0f, 0.0f, 1.0f);
	glm::vec3 camUp = glm::vec3(0.0f, 1.0f, 0.0f);

	Camera(glm::vec3 camPos, float camSpeed = 7.5f, float mouseSensitivity = .1f) {
		this->camPos = camPos;
		this->camSpeed = camSpeed;
		this->mouseSensitivity = mouseSensitivity;
	}
	Camera() {};

	void updateView() {
		view = glm::lookAt(camPos, camPos + camFront, camUp);
	}
	void processInput(GLFWwindow* window) {
		float speedAmplifier = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) ? 3.f : (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) ? 0.15 : 1.f);


		if (glfwGetKey(window, GLFW_KEY_W)) camPos += camSpeed * camFront * DT::time * speedAmplifier;
		if (glfwGetKey(window, GLFW_KEY_S)) camPos -= camSpeed * camFront * DT::time * speedAmplifier;
		if (glfwGetKey(window, GLFW_KEY_A)) camPos -= glm::normalize(glm::cross(camUp, camFront)) * camSpeed * DT::time * speedAmplifier;
		if (glfwGetKey(window, GLFW_KEY_D)) camPos += glm::normalize(glm::cross(camUp, camFront)) * camSpeed * DT::time * speedAmplifier;

		if (glfwGetKey(window, GLFW_KEY_W) || glfwGetKey(window, GLFW_KEY_A) || glfwGetKey(window, GLFW_KEY_S) || glfwGetKey(window, GLFW_KEY_D)){
			updateView();
		}
	}
	void processMouse(double xPos, double yPos) {
		if (firstMouse){
			lastX = xPos;
			lastY = yPos;
			firstMouse = false;
		}

		float xOffset = lastX - xPos;
		float yOffset = lastY - yPos;
		lastX = xPos;
		lastY = yPos;

		xOffset *= mouseSensitivity;
		yOffset *= mouseSensitivity;

		yaw += xOffset;
		pitch += yOffset;

		if (pitch >= 90.f) pitch = 89.9f;
		if (pitch <= -90.f) pitch = -89.9f;

		glm::vec3 dir;
		dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		dir.y = sin(glm::radians(pitch));
		dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		camFront = glm::normalize(dir);

		updateView();
	}
};

#endif