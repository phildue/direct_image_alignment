//
// Created by phil on 10.08.21.
//

#ifndef VSLAM_POSESE3PARAMETERIZATION_H
#define VSLAM_POSESE3PARAMETERIZATION_H


#include <ceres/local_parameterization.h>

namespace pd { namespace vision{
     //   https://github.com/stevenlovegrove/Sophus/blob/master/test/ceres/local_parameterization_se3.hpp
   class LocalParameterizationSE3 : public ceres::LocalParameterization {
   public:
       virtual ~LocalParameterizationSE3() {}

       /**
        * \brief SE3 plus operation for Ceres
        *
        * \f$ T\cdot\exp(\widehat{\delta}) \f$
        */
       virtual bool Plus(const double * T_raw, const double * delta_raw,
                         double * T_plus_delta_raw) const {
           const Eigen::Map<const Sophus::SE3d> T(T_raw);
           const Eigen::Map<const Eigen::Matrix<double,6,1> > delta(delta_raw);
           Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
           T_plus_delta = T * Sophus::SE3d::exp(delta);
           return true;
       }

       /**
        * \brief Jacobian of SE3 plus operation for Ceres
        *
        * \f$ \frac{\partial}{\partial \delta}T\cdot\exp(\widehat{\delta})|_{\delta=0} \f$
        */
       virtual bool ComputeJacobian(const double * T_raw, double * jacobian_raw)
       const {
           const Eigen::Map<const Sophus::SE3d> T(T_raw);
           Eigen::Map<Eigen::Matrix<double,6,7> > jacobian(jacobian_raw);
           jacobian = T.internalJacobian().transpose();
           return true;
       }

       virtual int GlobalSize() const {
           return Sophus::SE3d::num_parameters;
       }

       virtual int LocalSize() const {
           return Sophus::SE3d::DoF;
       }
   };



}}


#endif //VSLAM_POSESE3PARAMETERIZATION_H
