//
// Created by phil on 30.06.21.
//

#include "Frame.h"
namespace pd{
    namespace vision{


const Eigen::MatrixXd& Frame::grayImage(int level) const {
    return _grayImagePyramid[level];
}

Eigen::MatrixXd& Frame::grayImage(int level) {
    return _grayImagePyramid[level];
}

Eigen::Vector2d Frame::world2image(const Eigen::Vector3d &pWorld) const
{
    return camera2image(_pose * pWorld);
}
Eigen::Vector3d Frame::image2world(const Eigen::Vector2d &pImage, double depth) const
{
    return _poseInv * image2camera(pImage,depth);

}
Eigen::Vector2d Frame::camera2image(const Eigen::Vector3d &pCamera) const
{
    return camera()->camera2image(pCamera);
}
Eigen::Vector3d Frame::image2camera(const Eigen::Vector2d &pImage, double depth) const
{
    return camera()->image2camera(pImage,depth);
}

    }
}