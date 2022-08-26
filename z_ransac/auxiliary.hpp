//
// Created by liuxh on 22-7-12.
//
//#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int LeastSquaresMethod(const std::vector<int>& data_x, const std::vector<int>& data_y, double &k, double &b)
{
    if (data_x.size() != data_y.size())
    {
        return -1;
    }

    int num = data_x.size();
    double sxy = 0, sx = 0, sy = 0, sxx = 0;

    for (int i = 0; i < num; ++i)
    {
        sxy += data_x[i] * data_y[i];
        sx += data_x[i];
        sy += data_y[i];
        sxx += data_x[i] * data_x[i];
    }
    sxy *= num;
    sxx *= num;

    k = (sxy - sx * sy) / (sxx - sx * sx);
    b = sy / num - k * sx / num;

    return 0;
}


/* 预设求解的模型是线性模型 */
int Ransac(const std::vector<int>& data_x, const std::vector<int>& data_y, double &k, double &b)
{
    const int kIteration = 100;
    const int kRandomSampleInliner = 10;

    if (data_x.size() != data_y.size())
    {
        return -1;
    }
    int num = data_x.size();

    // fixed seed
    const unsigned seed = 123;
    // Mersenne Twister random engine:
    std::mt19937 urbg{ seed };
    // generate random ints ∈ [0, num - 1]
    std::uniform_int_distribution<int> distr{ 0, num - 1 };

    std::vector<int> sample_x(kRandomSampleInliner,0), sample_y(kRandomSampleInliner, 0);
    std::vector<int> inliner_num(kIteration, 0);   // 记录每轮随机采样求得直线后所有内群点数
    std::vector<std::pair<double,double> > tmp_kb(kIteration, std::pair<double,double>(0,0));   // 记录每轮随机采样求得直线的斜率、截距
    for (int i = 0; i < kIteration; ++i)
    {
        // 随机采样
        for (int j = 0; j < kRandomSampleInliner; ++j)
        {
            auto const idx = distr(urbg);
            std::cout << idx << std::endl;
            sample_x[j] = data_x[idx];
            sample_y[j] = data_y[idx];
        }

        // 利用LSM求解每轮采样点对应的直线
        double tmp_k = 0.0, tmp_b = 0.0;
        int ret = LeastSquaresMethod(sample_x, sample_y, tmp_k, tmp_b);
        if (0 != ret)
        {
            return ret;
        }

        // 获取求得的直线y=tmp_k*x+tmp_b"附近"内点总数，即将未采样的点带入求得的直线中确认是否未内群
        // 实现简单起见,将全部点带入一起确认内群点数量
        // 求点到直线的距离,tmp_k * x - y + tmp_b = 0
        // 设直线L的方程为Ax+By+C=0,点P(x0,y0)到L的距离为|A*x0 + B*y0 + C|/sqrt(A^2 + B^2)
        double kThreshDist = 2.0;  // 点距离直线2以内都认为是内群
        double dist = 0.0;
        for (int m = 0; m < num; ++m)
        {
            dist = fabs(tmp_k * data_x[m] - data_y[m] + tmp_b) / sqrt(tmp_k * tmp_k + 1 * 1);
            if (dist <= kThreshDist)
            {
                inliner_num[i]++;
            }
        }
        tmp_kb[i].first = tmp_k;
        tmp_kb[i].second = tmp_b;
    }
    auto it = std::max_element(inliner_num.begin(), inliner_num.end());
    int index = std::distance(inliner_num.begin(), inliner_num.begin());
    index = std::distance(inliner_num.begin(), it);

    k = tmp_kb[index].first;
    b = tmp_kb[index].second;

    return 0;
}
