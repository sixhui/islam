cmake_minimum_required(VERSION 2.8)
#project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

add_executable(orb_cv orb_cv.cpp ../Global.h.in feature.hpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp ../Global.h.in)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_2d3d pose_estimation_2d3d.cpp ../Global.h.in)
target_link_libraries(pose_estimation_2d3d ${OpenCV_LIBS} g2o_core g2o_stuff)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp ../Global.h.in)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} g2o_core g2o_stuff)