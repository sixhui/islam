//
// Created by liuxh on 22-8-20.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(){

    cv::Mat image = cv::imread("/home/liuxh/code/islam/z_opencv/hello/aar", -1);
    std::cout << image << std::endl;

    return 0;

}