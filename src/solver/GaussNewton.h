#ifndef VSLAM_LS_H__
#define VSLAM_LS_H__

#include <Eigen/Dense>


///
///
///
namespace pd{namespace vision{

    class LevenbergMarquardt{

        public:
        LevenbergMarquardt(std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd& )> computeResidual,
                std::function<bool(const Eigen::VectorXd&, Eigen::MatrixXd&)> computeJacobian,
                std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)> updateX,
                int nObservations,
                int nParameters,
                int maxIterations,
                double minStepSize,
                double minGradient
                );

        void solve(Eigen::VectorXd& x);
        void solve(Eigen::VectorXd& x, Eigen::VectorXd& chi2,Eigen::VectorXd &chi2pred, Eigen::VectorXd& lambda, Eigen::VectorXd& stepSize);

        double computeChi2(const Eigen::VectorXd& x,const Eigen::VectorXd& weights);
        private:
        std::function<bool(const Eigen::VectorXd& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights)> _computeResidual;
        std::function<bool(const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian)> _computeJacobian;
        std::function<bool(const Eigen::VectorXd& dx, Eigen::VectorXd& x)> _updateX;
        void computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights);
        const double _minGradient, _minStepSize;
        const int _maxIterations, _nObservations, _nParameters;
        const double _Lup = 5,_Ldown = 4; ///<Scalar to multiply lambda in case linearization was good/bad
    };

    class GaussNewton{

        public:
        GaussNewton(std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd& )> computeResidual,
                std::function<bool(const Eigen::VectorXd&, Eigen::MatrixXd&)> computeJacobian,
                std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)> updateX,
                int nObservations,
                int nParameters,
                double alpha,
                double minStepSize,
                int maxIterations
                );

        void solve(Eigen::VectorXd& x) const;
        void solve(Eigen::VectorXd& x, Eigen::VectorXd& chi2, Eigen::VectorXd& stepSize) const;

        double computeChi2(const Eigen::VectorXd& x,const Eigen::VectorXd& weights) const;
        private:
        std::function<bool(const Eigen::VectorXd& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights)> _computeResidual;
        std::function<bool(const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian)> _computeJacobian;
        std::function<bool(const Eigen::VectorXd& dx, Eigen::VectorXd& x)> _updateX;
        void computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights) const;
        const double _minStepSize, _alpha;
        const int _maxIterations, _nObservations, _nParameters;
    };
   
}}
#endif