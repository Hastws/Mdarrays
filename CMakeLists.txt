cmake_minimum_required(VERSION 3.5)
project(mdarray)

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
    add_executable(mdarray test/test_base_model.cpp ${SRC_FILES})
    add_executable(mlp_mnist test/mlp_mnist.cpp ${SRC_FILES})
    add_executable(cnn_mnist test/cnn_mnist.cpp ${SRC_FILES})
    add_executable(cnn_cifar_10 test/cnn_cifar_10.cpp ${SRC_FILES})
elseif (UNIX)
    message("Current platform unix.")
    add_subdirectory(./third_party/backward_cpp)
    add_executable(mdarray test/test_base_model.cpp ${SRC_FILES} ${BACKWARD_ENABLE})
    add_executable(mlp_mnist test/mlp_mnist.cpp ${SRC_FILES} ${BACKWARD_ENABLE})
    add_executable(cnn_mnist test/cnn_mnist.cpp ${SRC_FILES} ${BACKWARD_ENABLE})
    add_executable(cnn_cifar_10 test/cnn_cifar_10.cpp ${SRC_FILES} ${BACKWARD_ENABLE})
    add_backward(mdarray)
    add_backward(mlp_mnist)
    add_backward(cnn_mnist)
    add_backward(cnn_cifar_10)
endif ()

target_compile_definitions(mlp_mnist
        PRIVATE MLP_MNIST_TRAIN_IMAGES="${PROJECT_SOURCE_DIR}/data/mnist/train-images.idx3-ubyte"
        PRIVATE MLP_MNIST_TRAIN_LABELS="${PROJECT_SOURCE_DIR}/data/mnist/train-labels.idx1-ubyte"
        PRIVATE MLP_MNIST_TEST_IMAGES="${PROJECT_SOURCE_DIR}/data/mnist/t10k-images.idx3-ubyte"
        PRIVATE MLP_MNIST_TEST_LABELS="${PROJECT_SOURCE_DIR}/data/mnist/t10k-labels.idx1-ubyte"
)

target_compile_definitions(cnn_mnist PRIVATE
        PRIVATE MLP_MNIST_TRAIN_IMAGES="${PROJECT_SOURCE_DIR}/data/mnist/train-images.idx3-ubyte"
        PRIVATE MLP_MNIST_TRAIN_LABELS="${PROJECT_SOURCE_DIR}/data/mnist/train-labels.idx1-ubyte"
        PRIVATE MLP_MNIST_TEST_IMAGES="${PROJECT_SOURCE_DIR}/data/mnist/t10k-images.idx3-ubyte"
        PRIVATE MLP_MNIST_TEST_LABELS="${PROJECT_SOURCE_DIR}/data/mnist/t10k-labels.idx1-ubyte"
)

target_compile_definitions(cnn_cifar_10 PRIVATE
PRIVATE CIFAR_10_DATA_FOLDER="${PROJECT_SOURCE_DIR}/data/cifar_10"
)

target_compile_definitions(cnn_cifar_10 PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

target_link_libraries(multidimensional_arrays_lib memory_pool_lib)

target_link_libraries(mdarray multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(mlp_mnist multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(cnn_mnist multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(cnn_cifar_10 multidimensional_arrays_lib memory_pool_lib)