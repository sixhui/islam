cmake_minimum_required(VERSION 2.8)
#project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

add_executable(demo_opencv demo.cpp)
target_link_libraries(demo_opencv ${OpenCV_LIBS})

add_executable(opencv_version echo_opencv_version.cpp)
target_link_libraries(opencv_version ${OpenCV_LIBS})

add_executable(demo_opencv_sift demo_sift.cpp CvUtils.cpp)
target_link_libraries(demo_opencv_sift ${OpenCV_LIBS})