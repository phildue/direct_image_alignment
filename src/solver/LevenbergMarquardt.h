#ifndef VSLAM_LEVENBERG_MARQUARDT_H__
#define VSLAM_LEVENBERG_MARQUARDT_H__

#include <Eigen/Dense>
#include "solver.h"
namespace pd{namespace vision{

template<int nParameters>
    class LevenbergMarquardt : public Solver<nParameters>{
        using Mmxn = Eigen::Matrix<double, Eigen::Dynamic, nParameters>;
        public:
        LevenbergMarquardt(std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Eigen::VectorXd&, Eigen::VectorXd& )> computeResidual,
                std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Mmxn&)> computeJacobian,
                std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Eigen::Matrix<double, nParameters, 1>&)> updateX,
                int nObservations,
                int maxIterations,
                double minStepSize,
                double minGradient
                );

        void solve(Eigen::Matrix<double, nParameters, 1>& x) const override;
        void solve(Eigen::Matrix<double, nParameters, 1>& x, Eigen::VectorXd& chi2,Eigen::VectorXd &chi2pred, Eigen::VectorXd& lambda, Eigen::VectorXd& stepSize) const;

        private:
        std::function<bool(const Eigen::Matrix<double, nParameters, 1>& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights)> _computeResidual;
        std::function<bool(const Eigen::Matrix<double, nParameters, 1>& x, Mmxn& jacobian)> _computeJacobian;
        std::function<bool(const Eigen::Matrix<double, nParameters, 1>& dx, Eigen::Matrix<double, nParameters, 1>& x)> _updateX;
        void computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights) const;
        const double _minGradient, _minStepSize;
        const int _maxIterations, _nObservations, _nParameters;
        const double _Lup = 5,_Ldown = 4; ///<Scalar to multiply lambda in case linearization was good/bad
    };
}}
    #include "LevenbergMarquardt.hpp"

#endif