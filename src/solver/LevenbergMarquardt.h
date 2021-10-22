#ifndef VSLAM_LEVENBERG_MARQUARDT_H__
#define VSLAM_LEVENBERG_MARQUARDT_H__

#include <Eigen/Dense>

namespace pd{namespace vision{

template<int nParameters>
    class LevenbergMarquardt{
        typedef Eigen::Matrix<double, nParameters, 1> Vn;
        typedef Eigen::Matrix<double, Eigen::Dynamic, nParameters> Mmxn;
        public:
        LevenbergMarquardt(std::function<bool(const Vn&, Eigen::VectorXd&, Eigen::VectorXd& )> computeResidual,
                std::function<bool(const Vn&, Mmxn&)> computeJacobian,
                std::function<bool(const Vn&, Vn&)> updateX,
                int nObservations,
                int maxIterations,
                double minStepSize,
                double minGradient
                );

        void solve(Vn& x);
        void solve(Vn& x, Eigen::VectorXd& chi2,Eigen::VectorXd &chi2pred, Eigen::VectorXd& lambda, Eigen::VectorXd& stepSize);

        double computeChi2(const Vn& x,const Eigen::VectorXd& weights);
        private:
        std::function<bool(const Vn& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights)> _computeResidual;
        std::function<bool(const Vn& x, Mmxn& jacobian)> _computeJacobian;
        std::function<bool(const Vn& dx, Vn& x)> _updateX;
        void computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights);
        const double _minGradient, _minStepSize;
        const int _maxIterations, _nObservations, _nParameters;
        const double _Lup = 5,_Ldown = 4; ///<Scalar to multiply lambda in case linearization was good/bad
    };
}}
    #include "LevenbergMarquardt.hpp"

#endif