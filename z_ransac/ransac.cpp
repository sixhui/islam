//
// Created by liuxh on 22-7-12.
//
#include "auxiliary.hpp"

#include <random>
#include <vector>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    // fixed seed
    const unsigned seed = 123;
    // Mersenne Twister random engine:
    std::mt19937 urbg{ seed };
    // generate random ints ∈ [1,20]
    std::uniform_int_distribution<int> distr1{ 1, 20 };

    /* Step0,以k=3,b=2随机生成一些直线上的点 */
    const int kNum = 100;
    const int kK = 2;
    const int kB = 3;
    std::vector<int> data_x(kNum, 0);
    std::vector<int> data_y(kNum, 0);
    for (int i = 0; i < kNum; ++i)
    {
        data_x[i] = distr1(urbg);
        data_y[i] = kK * data_x[i] + kB;
    }

    /* Step1,给生成的点加噪声 */
    double const mu = 15.0;
    double const sigma = 5;
    auto norm = std::normal_distribution<double>{ mu,sigma };
    const int kNoiseNum = 30;
    for (int j = 0; j < kNoiseNum; ++j)
    {
        auto value = (int)norm(urbg);
        data_x.push_back(abs(value));
        value = (int)norm(urbg);
        data_y.push_back(abs(value));
    }

    /* Step2,执行ransac求解k和b */
    double ransac_k = 0.0, ransac_b = 0.0;
    int ret = Ransac(data_x, data_y, ransac_k, ransac_b);
    if (0 != ret)
    {
        return ret;
    }

    /* Step3,执行最小二乘法求解k和b,用于和ransac结果做对比 */
    double lsm_k = 0.0, lsm_b = 0.0;
    ret = LeastSquaresMethod(data_x, data_y, lsm_k, lsm_b);
    if (0 != ret)
    {
        return ret;
    }

    /* 可视化 */
    int width = *std::max_element(data_x.begin(), data_x.end()) + 2;
    int height = *std::max_element(data_y.begin(), data_y.end()) + 2;
    cv::Mat img(height, width, CV_8UC3);
    memset(img.data, 0, img.rows * img.cols * sizeof(uchar)*img.channels());
    for (int i = 0; i < data_x.size(); ++i)
    {
        if (i < kNum)  // 可视化y = 2x + 3上的点
        {
            img.at<cv::Vec3b>(data_y[i], data_x[i]) = cv::Vec3b(255, 0, 0);
        }
        else           // 可视化噪声点
        {
            img.at<cv::Vec3b>(data_y[i], data_x[i]) = cv::Vec3b(0, 255, 255);
        }
    }
    cv::imshow("data", img);
    // 可视化ransac求得的直线
    cv::Point pt1(0, ransac_k * 0 + ransac_b);
    cv::Point pt2(20, ransac_k * 20 + ransac_b);
    //cv::line(img, pt1, pt2, cv::Scalar(255, 0, 255), 1);
    // 可视化最小二乘法求得的直线
    pt1 = cv::Point(0, lsm_k * 0 + lsm_b);
    pt2 = cv::Point(20, lsm_k * 20 + lsm_b);
    cv::line(img, pt1, pt2, cv::Scalar(0, 0, 255), 1);
    cv::imshow("compare-result", img);
    cv::waitKey(0);

    return 0;
}

