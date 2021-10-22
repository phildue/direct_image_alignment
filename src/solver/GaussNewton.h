#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__

#include <Eigen/Dense>

namespace pd{namespace vision{

template<int nParameters>
      class GaussNewton{
        typedef Eigen::Matrix<double, nParameters, 1> Vn;
        typedef Eigen::Matrix<double, Eigen::Dynamic, nParameters> Mmxn;
 
        public:
        GaussNewton(std::function<bool(const Vn&, Eigen::VectorXd&, Eigen::VectorXd& )> computeResidual,
                std::function<bool(const Vn&, Mmxn&)> computeJacobian,
                std::function<bool(const Vn&, Vn&)> updateX,
                int nObservations,
                double alpha,
                double minStepSize,
                int maxIterations
                );

        void solve(Vn& x) const;
        void solve(Vn& x, Eigen::VectorXd& chi2, Eigen::VectorXd& stepSize) const;

        double computeChi2(const Eigen::VectorXd& x,const Eigen::VectorXd& weights) const;
        private:
        std::function<bool(const Vn& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights)> _computeResidual;
        std::function<bool(const Vn& x, Mmxn& jacobian)> _computeJacobian;
        std::function<bool(const Vn& dx, Vn& x)> _updateX;
        void computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights) const;
        const double _minStepSize, _alpha;
        const int _maxIterations, _nObservations, _nParameters;
    };
   
}}
#include "GaussNewton.hpp"
#endif