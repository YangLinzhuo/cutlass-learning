cmake_minimum_required(VERSION 3.22)

project(cute_learning VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find GoogleTest
find_package(GTest REQUIRED)

# Enable testing at the root level so that we could run all the tests at once
enable_testing()

# Set CUTLASS include directories
include_directories(cutlass/include)

file(GLOB SOURCE_FILES *.cu)

foreach(SRC_PATH ${SOURCE_FILES})
    get_filename_component(EXE_NAME ${SRC_PATH} NAME_WE)
    add_executable(${EXE_NAME} ${SRC_PATH})
    target_compile_options(${EXE_NAME} PRIVATE --expt-relaxed-constexpr)
endforeach(SRC_PATH ${SOURCE_FILES})
