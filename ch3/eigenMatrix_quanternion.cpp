//
// Created by liuxh on 22-8-9.
//
#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

int main(){

    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));

    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    std::cout << q.coeffs().transpose() << std::endl;

    Eigen::Quaterniond res = q * q.inverse();
    std::cout << res.coeffs().transpose() << std::endl;

    res = q * q;
    std::cout << res.coeffs().transpose() << std::endl;

}
