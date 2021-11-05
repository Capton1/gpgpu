cmake_minimum_required (VERSION 3.14)
project (GPGPU CXX)

include(FetchContent)

FetchContent_Declare(
        CLI11
        URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )

FetchContent_MakeAvailable(CLI11)
add_executable(main src/main.cpp)
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main PRIVATE CLI11::CLI11)

#find_package(PNG REQUIRED)






