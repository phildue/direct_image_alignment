//
// Created by phil on 02.07.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_MATH_H
#define DIRECT_IMAGE_ALIGNMENT_MATH_H

#include <Eigen/Dense>
namespace pd{
    namespace vision{
        namespace math{
            int bilinearInterpolation(const Eigen::MatrixXi& mat, double x, double y);
            double rmse(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2);
            Eigen::MatrixXi resize(const Eigen::MatrixXi& mat,double scale);
        }
    }
}
#endif //DIRECT_IMAGE_ALIGNMENT_MATH_H
