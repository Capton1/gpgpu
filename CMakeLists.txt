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

find_package(PNG REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(main src/main.cpp src/cpu.cpp)
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main PRIVATE CLI11::CLI11 PNG::PNG CLI11::CLI11 spdlog::spdlog)

#find_package(PNG REQUIRED)






