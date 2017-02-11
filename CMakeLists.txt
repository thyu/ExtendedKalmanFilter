cmake_minimum_required(VERSION 2.8)

project( DisplayImage )

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-O0 -g -Wall -Werror --std=c++11")

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable( ekf_test ekf_test.cpp )

target_link_libraries( ekf_test ${OpenCV_LIBS} )
