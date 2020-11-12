PROJECT(Vision)
# Include the Header files
include_directories(
    ${CMAKE_SOURCE_DIR}/lib_vision_gpu/src
)
include_directories(${CUDA_INCLUDE_DIRS})
# Include the libraries
link_directories(
    ${CMAKE_BINARY_DIR}/lib/lib_vision_gpu
)

aux_source_directory(./src SRC_SRC)
aux_source_directory(./src/def DEF_SRC)
aux_source_directory(./src/unix UNIX_SRC)

# message("the src is ${UNIX_SRC}" )

add_executable(Vision
    ${SRC_SRC}
    ${DEF_SRC}
    ${UNIX_SRC}
)

# link external library
target_link_libraries(Vision
    turbojpeg
    lib_vision_gpu
)
