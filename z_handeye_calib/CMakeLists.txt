cmake_minimum_required(VERSION 2.8)
#project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

add_executable(calib_handeye calib_handeye_opencv.cpp)
target_link_libraries(calib_handeye ${OpenCV_LIBS})

add_executable(calib_eye_in_hand calib_eye_in_hand.cpp)
target_link_libraries(calib_eye_in_hand ${OpenCV_LIBS})