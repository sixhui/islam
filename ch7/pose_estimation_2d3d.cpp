//
// Created by liuxh on 22-7-22.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>
#include "feature.hpp"
#include "../tools.hpp"
#include "../Global.h"
#include "pose_estimation_2d3d_GN.hpp"
#include "pose_estimation_2d3d_G2O.hpp"
using namespace std;
using namespace cv;


int main(int argc, char **argv) {
    cv::Mat                 img_1, img_2, d1;
    vector<cv::KeyPoint>    keypoints_1, keypoints_2;
    vector<cv::DMatch>      matches;
    vector<cv::Point3f>     pts_3d;
    vector<cv::Point2f>     pts_2d;
    Mat                     r, t, R;
    VecVector3d             pts_3d_eigen;
    VecVector2d             pts_2d_eigen;

    //-- 读取图像
    img_1 = imread(PATH_CH7_IMG1, CV_LOAD_IMAGE_COLOR);
    img_2 = imread(PATH_CH7_IMG2, CV_LOAD_IMAGE_COLOR);
    d1 = imread(PATH_CH7_DEP_IMG1, CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    assert(img_1.data && img_2.data && "Can not load images!");

    // 匹配角点
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;
    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
            continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    // EPnP - cv::API
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;

    // GN
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by gauss newton" << endl;
    Sophus::SE3d pose_gn;
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);

    // G2O
    cout << "calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    return 0;
}
