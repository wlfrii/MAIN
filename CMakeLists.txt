# Set cmake version
cmake_minimum_required(VERSION 3.0)

# Set the Project name and Version Number
project(Vision-AR VERSION 1.0)


SET(CMAKE_CXX_FLAGS "-std=c++17")

# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -s")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)


# Include the libraries
link_directories(
    /usr/lib
    /usr/local/lib
)


## Add sub
add_subdirectory(lib_vision_gpu)
add_subdirectory(Vision)