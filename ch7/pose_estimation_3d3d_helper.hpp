//
// Created by liuxh on 22-7-20.
//
#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>

/**
 * // 求质心 p1 p2
 * // 去质心坐标 q1i q2i
 * // W = sum{q1i*q2i^T}
 * // SVD on W
 * // R = U*V^T
 * // t = p1 - R*p2
 * @param pst1 输入图1的3D坐标
 * @param pst2 输入图2的3D坐标
 * @param R 输出旋转
 * @param t 输出平移
 */
void pose_estimation_3d3d_svd(const std::vector<cv::Point3f>& pst1,
                              const std::vector<cv::Point3f>& pst2,
                              cv::Mat& R, cv::Mat& t){
    if(pst1.size() != pst2.size()) throw std::invalid_argument("pst1.size() != pst2.size()");
    std::cout << pst1.size() << " " << pst2.size() << std::endl;
    int                         n = pst1.size();
    cv::Point3f                 p1, p2;         // 质心
    std::vector<cv::Point3f>    qs1(n), qs2(n);
    Eigen::Matrix3d             W = Eigen::Matrix3d::Zero(), U, V, R_;
    Eigen::Vector3d             t_;

    // 求质心 p1 p2
    for(int i = 0; i < n; ++i){
        p1 += pst1[i];
        p2 += pst2[i];
    }
    p1 /= n;
    p2 /= n;

    // 去质心坐标 q1i q2i
    for(int i = 0; i < n; ++i){
        qs1[i] = pst1[i] - p1;
        qs2[i] = pst2[i] - p2;
        W += Eigen::Vector3d(qs1[i].x, qs1[i].y, qs1[i].z) * Eigen::Vector3d(qs2[i].x, qs2[i].y, qs2[i].z).transpose();
    }

    // W = sum{q1i*q2i^T}
    std::cout << "W: " << W << std::endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();
    std::cout << U << std::endl;
    std::cout << V << std::endl;

    // R = U*V^T; t = p1 - R*p2
    R_ = U*V.transpose();
    if(R_.determinant() < 0){
        R_ = -R_;
    }

    // t = p1 - R*p2
    t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to CV
    R = (cv::Mat_<double>(3, 3) <<
            R_(0,0), R_(0,1), R_(0,2),
            R_(1,0), R_(1,1), R_(1,2),
            R_(2,0), R_(2,1), R_(2,2));
    t = (cv::Mat_<double>(3, 1) <<
            t_(0,0), t_(1,0), t_(2,0));
}