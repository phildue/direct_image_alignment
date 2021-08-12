//
// Created by phil on 10.08.21.
//

#ifndef VSLAM_COSTFUNCTION_H
#define VSLAM_COSTFUNCTION_H

#include <ceres/sized_cost_function.h>
#include <Eigen/Dense>

#include "core/Feature2D.h"
#include "core/Frame.h"
namespace pd { namespace vision{

    struct PhotometricLoss : ceres::SizedCostFunction<ceres::DYNAMIC,6>{
        Eigen::MatrixXd _jacobian;
        Eigen::MatrixXd _patchRef;
        const Eigen::Vector3d  _p3d;
        const Feature2D::ShConstPtr _ftRef;
        const Frame::ShConstPtr  _frameTarget;
        const std::uint32_t _patchSize;
        const std::uint32_t  _patchArea;
        const std::uint32_t  _patchSizeHalf;
        const std::uint32_t  _level;
        const double _scale;
        PhotometricLoss (Feature2D::ShConstPtr ftRef,
                         Frame::ShConstPtr frameTarget,
                         std::uint32_t patchSize,
                         std::uint32_t level
        );
        bool Evaluate(double const* const* parameters,double* residuals,double** jacobians) const ;
    };

    }}


#endif //VSLAM_COSTFUNCTION_H
