cmake_minimum_required(VERSION 3.25)

project(
    SIH26119_Solver
    VERSION 0.1.0
    DESCRIPTION "Indigenous GPU-Accelerated Mathematical Optimization Solver for SIH 2026 Problem Statement SIH26119"
    LANGUAGES CXX
)

# Enforce Modern C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Backend Architecture Options
option(ENABLE_CUDA "Enable CUDA GPU backend acceleration" OFF)
option(BUILD_TESTING "Enable solver test harness" ON)
option(BUILD_BENCHMARKS "Enable benchmark harness" OFF)

# Compiler Warnings Configuration
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()

# CUDA Backend Placeholder Support (Structured for future phases without redesign)
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA backend acceleration is ENABLED")
else()
    message(STATUS "CUDA backend acceleration is DISABLED (CPU foundation build)")
endif()

# Core Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Test Harness Integration via CTest
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests/unit)
endif()
