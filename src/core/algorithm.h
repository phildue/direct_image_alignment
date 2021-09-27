//
// Created by phil on 02.07.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H
#define DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H

#include <Eigen/Dense>
#include "types.h"

namespace pd{ namespace vision{ namespace algorithm{

    std::uint8_t bilinearInterpolation(const Image& mat, double x, double y);
    double rmse(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2);
    double sad(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2);
    Image resize(const Image& mat,double scale);

    /// Computes pixel wise gradient magnitude:  sqrt( (dI/dx)^2 + (dI/dy)^2
    /// \param image grayscale image
    /// \return gradient image
    Image gradient(const Image& image);
    Eigen::MatrixXi gradX(const Image& image);
    Eigen::MatrixXi gradY(const Image& image);

    /// Computes T01 from T0 and T1
    Sophus::SE3d computeRelativeTransform(const Sophus::SE3d& t0, const Sophus::SE3d& t1);

}}}
#endif //DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H
