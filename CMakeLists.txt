cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)
include(FetchContent)

FetchContent_Declare(
        CLI11
        URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
)


FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )

find_package(FreeImage REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules/)

message(${CMAKE_MODULE_PATH})
find_package(CUDA REQUIRED)

add_library(renderer)
target_compile_features(renderer PUBLIC cxx_std_14)
target_link_libraries(renderer PRIVATE CLI11::CLI11)
target_sources(renderer PRIVATE src/process.hpp src/process.cu)
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

FetchContent_MakeAvailable(CLI11)
add_executable(main src/main.cpp )
#add_executable(main src/main.cpp src/cpu.cpp src/utils.cpp src/image.cpp)
target_compile_features(main PUBLIC cxx_std_14)
target_link_libraries(main PRIVATE freeimage.so CLI11::CLI11 renderer)