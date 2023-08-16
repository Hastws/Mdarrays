cmake_minimum_required(VERSION 3.5)
project(multidimensional_arrays)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -Wall -fopenmp")

include_directories(include)
include_directories(third_party)

FILE(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

message(${PROJECT_SOURCE_DIR})

add_library(multidimensional_arrays_lib SHARED ${SRC_FILES})
add_library(memory_pool_lib SHARED ${PROJECT_SOURCE_DIR}/third_party/memory_pool/allocator.h
        ${PROJECT_SOURCE_DIR}/third_party/memory_pool/allocator.cpp)
if (WIN32)
    message("Current platform windows.")
    add_executable(multidimensional_arrays test/test_base_model.cpp ${SRC_FILES})
    add_executable(mda_train_mlp_simple test/train_mlp_simple.cpp ${SRC_FILES})
    add_executable(mda_train_cnn_simple test/train_cnn_simple.cpp ${SRC_FILES})
elseif (UNIX)
    message("Current platform unix.")
    add_subdirectory(./third_party/backward_cpp)
    add_executable(multidimensional_arrays test/test_base_model.cpp ${SRC_FILES} ${BACKWARD_ENABLE})
    add_executable(mda_train_mlp_simple test/train_mlp_simple.cpp ${SRC_FILES} ${BACKWARD_ENABLE})
    add_executable(mda_train_cnn_simple test/train_cnn_simple.cpp ${SRC_FILES} ${BACKWARD_ENABLE})
    add_backward(multidimensional_arrays)
    add_backward(mda_train_mlp_simple)
    add_backward(mda_train_cnn_simple)
endif ()

target_compile_definitions(multidimensional_arrays PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_compile_definitions(mda_train_mlp_simple PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_compile_definitions(mda_train_cnn_simple PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

target_link_libraries(multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(multidimensional_arrays multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(mda_train_mlp_simple multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(mda_train_cnn_simple multidimensional_arrays_lib memory_pool_lib)