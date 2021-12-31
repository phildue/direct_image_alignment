//
// Created by phil on 07.08.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_TYPES_H
#define DIRECT_IMAGE_ALIGNMENT_TYPES_H

#include <Eigen/Dense>
#include <sophus/se3.hpp>
typedef Eigen::Matrix<std::uint8_t ,Eigen::Dynamic,Eigen::Dynamic> Image;
typedef Eigen::Matrix<double ,Eigen::Dynamic,Eigen::Dynamic> DepthMap;

typedef std::uint64_t Timestamp;
typedef Sophus::SE3d SE3d;
namespace Eigen{
    typedef Eigen::Matrix<double,6,1> Vector6d;
}
#endif //DIRECT_IMAGE_ALIGNMENT_TYPES_H
