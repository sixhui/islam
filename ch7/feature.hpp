//
// Created by liuxh on 22-7-20.
//
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 *
 * @param img1 图像1
 * @param img2 图像2
 * @param keypoints_1 图像1提取的角点
 * @param keypoints_2 图像2提取的角点
 * @param matches 两个图像角点的匹配结果
 */
void find_feature_matches(cv::Mat& img1, cv::Mat& img2,
                          std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches){
    // 初始化
    cv::Mat                 descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches_raw;
    double                  min_dist = 10000, max_dist = 0;
    double                  threshold = 30.0;

    cv::Ptr<cv::FeatureDetector>        f_detector  = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor>    d_extractor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher>      matcher     = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // 检测角点
    f_detector->detect(img1, keypoints_1);
    f_detector->detect(img2, keypoints_2);

    // 计算描述子
    d_extractor->compute(img1, keypoints_1, descriptors_1);
    d_extractor->compute(img2, keypoints_2, descriptors_2);

    // 匹配
    matcher->match(descriptors_1, descriptors_2, matches_raw);
    std::cout << "key points : " << keypoints_1.size() << " " << keypoints_2.size() << std::endl;
    std::cout << "descriptors: " << descriptors_1.size() << " " << descriptors_2.size() << std::endl;
    std::cout << "matches    : " << matches_raw.size() << std::endl;

    // 筛选 - 匹配点对的距离
    for(const auto m: matches_raw){
        if(m.distance < min_dist) min_dist = m.distance;
        if(m.distance > max_dist) max_dist = m.distance;
    }
    std::cout << "max dist: " << max_dist << "-" << "min dist: " << min_dist << std::endl;

    threshold = std::max(threshold, min_dist * 2);
    for(const auto m: matches_raw){
        if(m.distance < threshold) matches.push_back(m);
    }
}

