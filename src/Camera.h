//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_CAMERA_H
#define DIRECT_IMAGE_ALIGNMENT_CAMERA_H

#include <Eigen/Dense>
#include <memory>
namespace pd{
    namespace vision {

        class Camera {
        public:
            using ConstShPtr = std::shared_ptr<const Camera>;
            using ShPtr = std::shared_ptr<Camera>;

            Camera(double f, double cx, double cy);

            Eigen::Vector2d camera2image(const Eigen::Vector3d &pCamera) const;
            Eigen::Vector3d image2camera(const Eigen::Vector2d &pImage, double depth = 1.0) const;
            Eigen::Vector3d image2ray(const Eigen::Vector2d &pImage) const;

            Eigen::Matrix<double,2,6> J_xyz2uv(const Eigen::Vector3d &pCamera) const;
            const double& focalLength() const {return _focalLength;}
        private:
            double _focalLength;
            Eigen::Matrix<double, 3, 3 > _K; //< Intrinsic camera matrix
            Eigen::Matrix<double, 3, 3 > _Kinv; //< Intrinsic camera matrix inverted
        };
    }}

#endif //DIRECT_IMAGE_ALIGNMENT_CAMERA_H
