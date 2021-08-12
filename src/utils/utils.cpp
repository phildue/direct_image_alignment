//
// Created by phil on 08.08.21.
//
#include "utils.h"
#include "Exceptions.h"

namespace pd{ namespace vision {
        void utils::throw_if_nan(const Eigen::MatrixXd &mat, const std::string &msg, std::shared_ptr<const Feature2D> ft)
        {
            if (std::isnan(mat.norm()))
            {
                std::stringstream ss;
                ss << mat;
                throw pd::Exception(msg + " for feature ["+ std::to_string(ft->id()) +"] at (" + std::to_string(ft->position().x())+ ","+ std::to_string(ft->position().y()) +  ") contains nan: \n" + ss.str());
            }
        }
    }}