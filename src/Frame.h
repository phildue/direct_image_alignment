//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_FRAME_H
#define DIRECT_IMAGE_ALIGNMENT_FRAME_H

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include "Feature2D.h"
#include "Camera.h"

namespace pd{
    namespace vision {

        class Frame {
        public:
            using ShPtr = std::shared_ptr<Frame>;
            using ShConstPtr = std::shared_ptr<Frame>;
            const Eigen::MatrixXd& grayImage(int level = 0) const;
            Eigen::MatrixXd& grayImage(int level = 0);
            Eigen::Vector2d world2image(const Eigen::Vector3d &pWorld) const;
            Eigen::Vector3d image2world(const Eigen::Vector2d &pImage, double depth = 1.0) const;
            Eigen::Vector2d camera2image(const Eigen::Vector3d &pCamera) const;
            Eigen::Vector3d image2camera(const Eigen::Vector2d &pImage, double depth = 1.0) const;
            Camera::ConstShPtr camera() const { return _camera;};
            const std::vector<Feature2D::ShConstPtr>& features() const { return _features;}
            std::vector<Feature2D::ShConstPtr>& features() { return _features;}
            Sophus::SE3d& pose() { return _pose;}
            const Sophus::SE3d& pose() const { return _pose;}
        private:
            std::vector<Feature2D::ShConstPtr> _features;
            std::vector<Eigen::MatrixXd> _grayImagePyramid;
            Camera::ConstShPtr _camera;
            Sophus::SE3d _pose,_poseInv;
        };
    }}

#endif //MYLIBRARY_FRAME_H
