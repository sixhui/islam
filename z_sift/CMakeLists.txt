cmake_minimum_required(VERSION 2.8)
#project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

add_executable(sift_main main.cpp myDisplay.cpp myMatch.cpp mySift.cpp)
target_link_libraries(sift_main ${OpenCV_LIBS})
