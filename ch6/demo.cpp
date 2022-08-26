//
// Created by liuxh on 22-7-21.
//
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "../Global.h"

/**
 * y = exp(a)x^2 + bx + c
 * @return
 */

template <typename T>
void write_vector_2_txt(const std::vector<T>& vec, std::string path){
    std::ofstream out_file;
    out_file.open(path);
    for(const auto e: vec){
        out_file << e << std::endl;
    }
    out_file.close();
}

int main(){
    std::vector<double> x_data, y_data;      // 数据

    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    std::ofstream out_file_x, out_file_y;
    out_file_x.open(PATH_CH6_DATA_X);
    out_file_y.open(PATH_CH6_DATA_Y);
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
        out_file_x << x_data.back() << std::endl;
        out_file_y << y_data.back() << std::endl;
    }
    for(auto e: x_data) std::cout << e << std::endl;
    for(auto e: y_data) std::cout << e << std::endl;
    out_file_x.close();
    out_file_y.close();
    std::cout << x_data[0] << " " << y_data[0] << std::endl;

    read_txt_2_vector(PATH_CH6_DATA_X, x_data);
    read_txt_2_vector(PATH_CH6_DATA_Y, y_data);
    show(y_data);

//    std::ofstream out_file;
//    out_file.open(PATH_CH6_DATA_X);
//
//    out_file << "bbb" << std::endl;
//
//    out_file.close();

}