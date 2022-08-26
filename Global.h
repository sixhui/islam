#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>

#define PROJECT_PATH        "/home/liuxh/code/islam/"

#define PATH_CH6            PROJECT_PATH"ch6/"
#define PATH_CH6_DATA_X     PATH_CH6"data_x.txt"
#define PATH_CH6_DATA_Y     PATH_CH6"data_y.txt"

#define PATH_CH7            PROJECT_PATH"ch7/"
#define PATH_CH7_IMG1       PATH_CH7"1.png"
#define PATH_CH7_IMG2       PATH_CH7"2.png"
#define PATH_CH7_DEP_IMG1   PATH_CH7"1_depth.png"
#define PATH_CH7_DEP_IMG2   PATH_CH7"2_depth.png"

#define PATH_ZZY            PROJECT_PATH"zzy/"
#define PATH_ZZY_IMGS_PTN   PATH_ZZY"images/*.jpg"

extern const cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);


std::string path_concat(std::string pre, std::string file){
    pre += file;
    return pre;
}

void read_txt_2_vector(std::string path, std::vector<double>& vec){
    std::ifstream file(path);
    double tmp;
    while(!file.eof()){
        file >> tmp;
        vec.push_back(tmp);
    }
}

template <typename T>
void show(const std::vector<T>& vec){
    for(auto& e: vec) std::cout << e << std::endl;
}
