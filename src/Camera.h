//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_CAMERA_H
#define DIRECT_IMAGE_ALIGNMENT_CAMERA_H

#include <Eigen/Dense>
namespace pd{
    namespace vision {

        class Camera {
        public:
            Eigen::Vector2d camera2image(const Eigen::Vector3d &pCamera) const;
            Eigen::Vector3d image2camera(const Eigen::Vector2d &pImage, double depth = 1.0) const;
            Eigen::Vector3d image2ray(const Eigen::Vector2d &pImage) const;

        private:
            Eigen::Matrix3d _K; //< Intrinsic camera matrix
            Eigen::Matrix3d _Kinv; //< Intrinsic camera matrix inverted
        };
    }}

#endif //DIRECT_IMAGE_ALIGNMENT_CAMERA_H
