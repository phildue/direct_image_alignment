//
// Created by phil on 08.08.21.
//

#ifndef VSLAM_UTILS_H
#define VSLAM_UTILS_H

#include <string>
#include <Eigen/Dense>

#include "Exceptions.h"
#include "Log.h"
#include "core/Feature2D.h"
namespace pd{ namespace vision { namespace utils{
    void throw_if_nan(const Eigen::MatrixXd& mat,const std::string& msg, std::shared_ptr<const Feature2D> ft);
}}}
#endif //SRC_UTILS_H
