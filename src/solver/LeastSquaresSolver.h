#ifndef VSLAM_LS_H__
#define VSLAM_LS_H__

#include <Eigen/Dense>

namespace pd{namespace vision{

    class LeastSquaresSolver{

        public:
        LeastSquaresSolver(std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::MatrixXd& )> computeResidual,
                std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&)> computeJacobian,
                std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&)> updateX,
                int nObservations,
                int nParameters,
                double lambda0,
                double minStepSize,
                int maxIterations
                );
        void solve(Eigen::MatrixXd& x);
        void solve(Eigen::MatrixXd& x, Eigen::VectorXd& chi2, Eigen::VectorXd& lambda);


        double computeChi2(const Eigen::MatrixXd& x,const Eigen::MatrixXd& weights);
        private:
        std::function<bool(const Eigen::MatrixXd& x, Eigen::MatrixXd& residual, Eigen::MatrixXd& weights)> _computeResidual;
        std::function<bool(const Eigen::MatrixXd& x, Eigen::MatrixXd& jacobian)> _computeJacobian;
        std::function<bool(const Eigen::MatrixXd& dx, Eigen::MatrixXd& x)> _updateX;
        const double _lambda0, _minStepSize;
        const int _maxIterations, _nObservations, _nParameters;
    };
}}
#endif