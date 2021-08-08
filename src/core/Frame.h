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

            explicit  Frame(const Eigen::MatrixXi& grayImage, Camera::ConstShPtr camera,uint32_t levels = 1, const Sophus::SE3d& pose = Sophus::SE3d());
            void addFeature(Feature2D::ShConstPtr ft);
            void removeFeatures();

            const Eigen::MatrixXi& grayImage(int level = 0) const;
            Eigen::MatrixXi& grayImage(int level = 0);

            const Eigen::MatrixXi& gradientImage(int level = 0) const;
            Eigen::MatrixXi& gradientImage(int level = 0);

            Eigen::Vector2d world2image(const Eigen::Vector3d &pWorld) const;
            Eigen::Vector3d image2world(const Eigen::Vector2d &pImage, double depth = 1.0) const;
            Eigen::Vector2d camera2image(const Eigen::Vector3d &pCamera) const;
            Eigen::Vector3d image2camera(const Eigen::Vector2d &pImage, double depth = 1.0) const;
            Camera::ConstShPtr camera() const { return _camera;};
            const std::vector<Feature2D::ShConstPtr>& features() const { return _features;}
            uint32_t nObservedPoints() const;
            Sophus::SE3d& pose() { return _pose;}
            const Sophus::SE3d& pose() const { return _pose;}
            uint32_t width(uint32_t level = 0) const;
            uint32_t height(uint32_t level = 0) const;
            bool isVisible(const Eigen::Vector2d& pImage, double border, uint32_t level = 0) const;
            int levels() const { return _grayImagePyramid.size();};
        private:
            std::vector<Feature2D::ShConstPtr> _features;
            std::vector<Eigen::MatrixXi> _grayImagePyramid;
            std::vector<Eigen::MatrixXi> _gradientImagePyramid;
            Camera::ConstShPtr _camera;
            Sophus::SE3d _pose,_poseInv;
        };
    }}

#endif //MYLIBRARY_FRAME_H
