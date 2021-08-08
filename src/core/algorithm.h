//
// Created by phil on 02.07.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H
#define DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H

#include <Eigen/Dense>
#include "types.h"

namespace pd{ namespace vision{ namespace algorithm{

    int bilinearInterpolation(const Eigen::MatrixXi& mat, double x, double y);
    double rmse(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2);
    Eigen::MatrixXi resize(const Eigen::MatrixXi& mat,double scale);

    /// Computes pixel wise gradient magnitude:  sqrt( (dI/dx)^2 + (dI/dy)^2
    /// \param image grayscale image
    /// \return gradient image
    Image gradient(const Image& image);
    Image gradX(const Image& image);
    Image gradY(const Image& image);

}}}
#endif //DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H
