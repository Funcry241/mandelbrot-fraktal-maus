// src/input/callbacks.hpp

#pragma once

#include <GLFW/glfw3.h>

// Callback-Funktionen (werden intern registriert)
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);

// Registriert alle Callback-Funktionen bei GLFW
void init_callbacks(GLFWwindow* window);
