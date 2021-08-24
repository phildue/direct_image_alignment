//
// Created by phil on 24.08.21.
//

#ifndef VSLAM_SE3PARAMETERIZATION_H
#define VSLAM_SE3PARAMETERIZATION_H

class SE3qtParam : public ceres::LocalParameterization {
public:
    virtual ~SE3qtParam() {}

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

class SE3atParam : public ceres::LocalParameterization {
public:
    virtual ~SE3atParam() {}

    /**
     * \brief SE3 plus operation for Ceres
     *
     * \f$ T\cdot\exp(\widehat{\delta}) \f$
     */
    virtual bool Plus(const double * T_raw, const double * delta_raw,
                      double * T_plus_delta_raw) const {

        const Eigen::Map<const Eigen::Matrix<double,6,1>> v(T_raw);
        const auto T =  Sophus::SE3d::exp(v);
        const Eigen::Map<const Eigen::Matrix<double,6,1> > delta(delta_raw);
        Eigen::Map<Eigen::Matrix<double,6,1>> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = (Sophus::SE3d::exp(delta)*T).log();
        VLOG(5) << "Update: \n" << v.transpose() << " + \n" << delta.transpose() << "\n =" << T_plus_delta.transpose();
        return true;
    }

    /**
     * \brief Jacobian of SE3 plus operation for Ceres
     *
     * \f$ \frac{\partial}{\partial \delta}T\cdot\exp(\widehat{\delta})|_{\delta=0} \f$
     */
    virtual bool ComputeJacobian(const double * T_raw, double * jacobian_raw)
    const {
        Eigen::Map<Eigen::Matrix<double, Sophus::SE3d::DoF, Sophus::SE3d::DoF, Eigen::RowMajor> > J(jacobian_raw);
        J.setIdentity();//we can do this because this is just to map from the global to the local Jacobian??
        return true;
    }

    virtual int GlobalSize() const {
        return Sophus::SE3d::DoF;
    }

    virtual int LocalSize() const {
        return Sophus::SE3d::DoF;
    }
};


#endif //VSLAM_SE3PARAMETERIZATION_H
