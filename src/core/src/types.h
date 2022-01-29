//
// Created by phil on 07.08.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_TYPES_H
#define DIRECT_IMAGE_ALIGNMENT_TYPES_H

#include <Eigen/Dense>
#include <sophus/se3.hpp>


namespace Eigen{
    typedef Eigen::Matrix<double,6,1> Vector6d;
}

namespace pd::vision{
    typedef Eigen::Matrix<std::uint8_t ,Eigen::Dynamic,Eigen::Dynamic> Image;
    typedef Eigen::Matrix<double ,Eigen::Dynamic,Eigen::Dynamic> DepthMap;
    typedef std::uint64_t Timestamp;
    typedef Sophus::SE3d SE3d;
    typedef Eigen::Matrix<std::uint8_t ,Eigen::Dynamic,Eigen::Dynamic> MatUi8;
    typedef Eigen::Matrix<int ,Eigen::Dynamic,Eigen::Dynamic> MatI;
    typedef Eigen::Matrix<double ,Eigen::Dynamic,Eigen::Dynamic> MatD;
    typedef Eigen::Matrix<float ,Eigen::Dynamic,Eigen::Dynamic> MatF;
    typedef Eigen::Matrix<double ,3,3> Mat3D;
    typedef Eigen::Matrix<double ,2,2> Mat2D;

    typedef Eigen::VectorXd VecD;
    typedef Eigen::Vector2d Vec2D;
    typedef Eigen::Vector3d Vec3D;
    typedef Eigen::Vector6d Vec6D;


}

#endif //DIRECT_IMAGE_ALIGNMENT_TYPES_H
