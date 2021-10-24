#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__

#include <Eigen/Dense>
#include "solver.h"

namespace pd{namespace vision{

template<int nParameters>
      class GaussNewton : public Solver<nParameters>{
        typedef Eigen::Matrix<double, Eigen::Dynamic, nParameters> Mmxn;
 
        public:
        GaussNewton(std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Eigen::VectorXd&, Eigen::VectorXd& )> computeResidual,
                std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Mmxn&)> computeJacobian,
                std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Eigen::Matrix<double, nParameters, 1>&)> updateX,
                int nObservations,
                double alpha,
                double minStepSize,
                int maxIterations
                );

        void solve(Eigen::Matrix<double, nParameters, 1>& x) const override;
        void solve(Eigen::Matrix<double, nParameters, 1>& x, Eigen::VectorXd& chi2, Eigen::VectorXd& stepSize) const;

        private:
        std::function<bool(const Eigen::Matrix<double, nParameters, 1>& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights)> _computeResidual;
        std::function<bool(const Eigen::Matrix<double, nParameters, 1>& x, Mmxn& jacobian)> _computeJacobian;
        std::function<bool(const Eigen::Matrix<double, nParameters, 1>& dx, Eigen::Matrix<double, nParameters, 1>& x)> _updateX;
        void computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights) const;
        const double _minStepSize, _alpha;
        const int _maxIterations, _nObservations, _nParameters;
    };
   
}}
#include "GaussNewton.hpp"
#endif