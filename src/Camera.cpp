//
// Created by phil on 30.06.21.
//

#include "Camera.h"
namespace pd{
    namespace vision {

        Eigen::Vector2d Camera::camera2image(const Eigen::Vector3d &pWorld) const {
            auto pProj = _K * pWorld;
            return {pProj.x()/pProj.z(), pProj.y()/pProj.z()};
        }

        Eigen::Vector3d Camera::image2camera(const Eigen::Vector2d &pImage, double depth) const {
            return image2ray(pImage) * depth;
        }
        Eigen::Vector3d Camera::image2ray(const Eigen::Vector2d &pImage) const {
            return _Kinv * Eigen::Vector3d({pImage.x(),pImage.y(),1});
        }
}