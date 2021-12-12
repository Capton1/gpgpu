cmake_minimum_required (VERSION 3.14)
project (GPGPU CXX)

include(FetchContent)

FetchContent_Declare(
        CLI11
        URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
)

FetchContent_Declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
)

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )

find_package(FreeImagePlus REQUIRED)
include_directories(${FREEIMAGEPLUS_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

#src/cpu.cpp src/utils.cpp src/image.cpp
add_executable(main src/main.cpp )
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main PRIVATE CLI11::CLI11 freeimageplus.so CLI11::CLI11 spdlog::spdlog)








