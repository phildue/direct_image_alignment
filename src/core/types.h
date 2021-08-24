//
// Created by phil on 07.08.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_TYPES_H
#define DIRECT_IMAGE_ALIGNMENT_TYPES_H

#include <Eigen/Dense>
#include <sophus/se3.hpp>
typedef Eigen::Matrix<std::uint8_t ,Eigen::Dynamic,Eigen::Dynamic> Image;
typedef std::uint64_t Timestamp;
typedef Sophus::SE3d SE3d;

#endif //DIRECT_IMAGE_ALIGNMENT_TYPES_H
