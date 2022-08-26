//
// Created by liuxh on 22-7-20.
//
#ifndef __TOOLS_
#define __TOOLS_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

/**
 * 像素坐标转相机归一化坐标
 * @param p 像素坐标 (u, v)
 * @param K 内参
 * @return
 */
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;



#endif