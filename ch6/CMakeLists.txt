cmake_minimum_required(VERSION 2.8)
#project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# 添加Eigen头文件
add_executable(CurveFitting_G2O CurveFitting_G2O.cpp)
target_link_libraries(CurveFitting_G2O ${OpenCV_LIBS} g2o_core g2o_stuff)

add_executable(CurveFitting_GN CurveFitting_GN.cpp)
target_link_libraries(CurveFitting_GN ${OpenCV_LIBS})

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS} g2o_core g2o_stuff)