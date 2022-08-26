cmake_minimum_required(VERSION 2.8)
project(islam)

set(CMAKE_CXX_STANDARD 14)
set(PROJECT_PATH ${PROJECT_SOURCE_DIR})
#set(OPENCV_DIR "/usr/local/opencv_3.4.16/share/OpenCV")
set(OPENCV_DIR "/usr/local/opencv_4.5.5/share/OpenCV")


configure_file(${PROJECT_SOURCE_DIR}/Global.h.in ${PROJECT_SOURCE_DIR}/Global.h)

# g2o
list(APPEND CMAKE_MODULE_PATH /home/liuxh/Downloads/g2o/cmake_modules)


find_package(OpenCV 4 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        "/usr/local/include/sophus"
        "/usr/include/glog"
#        "${PROJECT_SOURCE_DIR}/Global.h "
)

add_executable(helloSLAM helloSLAM.cpp)

add_executable(vio_ch1 VIO/VIO_ch1.cpp)
#target_link_libraries(useSophus Sophus::Sophus)

#add_executable(orb_cv ch7/orb_cv.cpp)
#target_link_libraries(orb_cv ${OpenCV_LIBS})
#add_executable(triangulation ch7/triangulation.cpp)
#target_link_libraries(triangulation ${OpenCV_LIBS})

#add_executable(eigenMatrix ch3/eigenMatrix.cpp)

add_subdirectory(ch3)
add_subdirectory(ch6)
add_subdirectory(ch7)
add_subdirectory(ch12)
add_subdirectory(z_handeye_calib)
add_subdirectory(z_cholesky)
add_subdirectory(z_ransac)
add_subdirectory(z_sift)
add_subdirectory(z_opencv)



add_executable(calib_zzy zzy/main.cpp Global.h.in zzy/class_camera_calibrator.hpp zzy/class_nonlinear_optimizer.hpp)
target_link_libraries(calib_zzy gflags glog ${OpenCV_LIBS}  ${CERES_LIBRARIES})

add_executable(calib_zzy_new zzy_new/main.cpp Global.h.in zzy_new/class_camera_calibrator.hpp zzy_new/class_nonlinear_optimizer.hpp)
target_link_libraries(calib_zzy_new gflags glog ${OpenCV_LIBS}  ${CERES_LIBRARIES})

