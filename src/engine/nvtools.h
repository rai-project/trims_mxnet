#pragma once

#include <chrono>
#include <iostream>

#define SET_CLOCK(t0)                                                          \
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

#define TIME_DIFF(t1, t0)                                                      \
  (std::chrono::duration_cast<std::chrono::duration<double>>((t1) - (t0))      \
       .count())

#define PRINT_CLOCK(msg, t1, t0) std::cerr << msg << TIME_DIFF(t1, t0) << endl;

#include <nvToolsExt.h>

static const uint32_t colors[] = {0xFFFFFF00,  // Yellow
                                  0xFFFF00FF,  // Fuchsia
                                  0xFFFF0000,  // Red
                                  0xFFC0C0C0,  // Silver
                                  0xFF808080,  // Gray
                                  0xFF808000,  // Olive
                                  0xFF800080,  // Purple
                                  0xFF800000,  // Maroon
                                  0xFF00FFFF,  // Aqua
                                  0xFF00FF00,  // Lime
                                  0xFF008080,  // Teal
                                  0xFF008000,  // Green
                                  0xFF0000FF,  // Blue
                                  0xFF000080}; // Navy
static const int num_colors = sizeof(colors) / sizeof(uint32_t);

