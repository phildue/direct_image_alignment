//
// Created by phil on 02.07.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_MATH_H
#define DIRECT_IMAGE_ALIGNMENT_MATH_H

#include <Eigen/Dense>
namespace pd{
    namespace vision{
        namespace math{
            double bilinearInterpolation(const Eigen::MatrixXd& mat, double x, double y);
            double rmse(const Eigen::MatrixXd& patch1, const Eigen::MatrixXd& patch2);
            Eigen::MatrixXd resize(const Eigen::MatrixXd& mat,double scale);
        }
    }
}
#endif //DIRECT_IMAGE_ALIGNMENT_MATH_H
