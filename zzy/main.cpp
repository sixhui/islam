#include <gflags/gflags.h>
#include "class_camera_calibrator.hpp"
#include "../Global.h"
using namespace std;

int main()
{
	FLAGS_log_dir = "./";
	FLAGS_colorlogtostderr = true;
	google::InitGoogleLogging("calibrator");
	google::LogToStderr();

	vector<cv::String> vpath_images;            // 全部图像的绝对路径
	cv::glob(PATH_ZZY_IMGS_PTN, vpath_images);
	vector<cv::Mat> vec_mat;                    // 全部图像
	for (const auto &path:vpath_images){
		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		vec_mat.push_back(img);
	}

	CameraCalibrator m;
	Eigen::Matrix3d camera_matrix;              // 相机标定得到的内参矩阵，每个相机一个
	Eigen::VectorXd k;                          //
	vector<Eigen::MatrixXd> vec_extrinsics;     // 相机标定得到的外参矩阵，每张图片一个

	m.set_input(vec_mat, cv::Size{ 9,6 });
	m.get_result(camera_matrix,k,vec_extrinsics);

	std::cout << "camera_matrix:\n" << camera_matrix << std::endl;
	std::cout << "k:\n" << k << std::endl;
	//for (int i=0;i<vec_extrinsics.size();++i)
	//{
	//	LOG(INFO) << "vec_extrinsics["<<i<<"]:\n" << vec_extrinsics[i] << std::endl;
	//}
	Eigen::Matrix3d opencv_camera_matrix;
	opencv_camera_matrix << 532.79536563, 0., 342.4582516,
		0, 532.91928339, 233.90060514,
		0, 0, 1;
	std::cout << "opencv calibrateCamera api result:\n" << opencv_camera_matrix << std::endl;
	std::cin.get();
}