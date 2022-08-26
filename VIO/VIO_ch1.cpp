//
// Created by liuxh on 22-6-20.
//


#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

Matrix3d hat(const Vector3d& v){
    Matrix3d m;
    m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return m;
}

int main(int argc, char **argv) {

    // initial pose
    AngleAxisd  pose_r_vector   = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1));
    Matrix3d    R               = pose_r_vector.toRotationMatrix();
    Quaterniond q               = Quaterniond(pose_r_vector);

    // w - angular velocity
    Vector3d    w(0.01, 0.02, 0.03);

    // Rotation Matrix
    double      theta           = w.norm();
    Vector3d    a               = w / theta;
    Matrix3d    R_new           =
            cos(theta) * Eigen::Matrix3d::Identity()
            + (1 - cos(theta)) * a * a.transpose()
            + sin(theta) * hat(a);                  // Rodrigues' formula

    Matrix3d    R_res           = R * R_new;

    // Quaternion
    Quaterniond q_new(1, w[0] / 2, w[1] / 2, w[2] / 2);
    Quaterniond q_res           = q * q_new;
    q_res                       = q_res.normalized();

    // Diff
    cout << q_res.toRotationMatrix() - R_res << endl;


    return 0;
}
