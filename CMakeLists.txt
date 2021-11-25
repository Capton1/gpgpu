cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

include(FetchContent)

FetchContent_Declare(
        CLI11
        URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(OpenCV_DIR ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules/)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules/)

message(${CMAKE_MODULE_PATH})
find_package(CUDA REQUIRED)
set("OpenCV_DIR" "/usr/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(renderer)
target_compile_features(renderer PUBLIC cxx_std_14)
target_link_libraries(renderer PRIVATE CLI11::CLI11)
target_sources(renderer PRIVATE src/process.hpp src/process.cu)
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

FetchContent_MakeAvailable(CLI11)
add_executable(main src/main.cpp)
target_compile_features(main PUBLIC cxx_std_14)
target_link_libraries(main PRIVATE CLI11::CLI11 ${OpenCV_LIBS} renderer)

#find_package(PNG REQUIRED)