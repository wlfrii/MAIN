PROJECT(lib_vision_gpu)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib_vision_gpu
)

find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
    include_directories(${CUDA_INCLUDE_DIRS})

    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
    SET(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;-rdc=true;)
    # --disable-warnings

    message("the cu file is ### ${CUDA_NVCC_FLAGS}")
    set(CUDA_SEPARABLE_COMPILATION ON)

    link_directories($ENV{CUDA_PATH}/lib/)
else(${CUDA_FOUND})
    message(STATUS "CUDA not found!")
endif(${CUDA_FOUND})


file(GLOB_RECURSE SRC_CU "./src/*.cu")
file(GLOB_RECURSE SRC_CPP "./src/*.cpp")
 message("the cu file is ### ${SRC_CU}")
# message("the cpp file is ### ${SRC_CPP}")

cuda_add_library(lib_vision_gpu STATIC
    ${SRC_CPP}
    ${SRC_CU}
)

target_link_libraries(lib_vision_gpu
    cudadevrt
    cudart_static
)