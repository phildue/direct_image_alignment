#ifndef VSLAM_LEVENBERG_MARQUARDT_H__
#define VSLAM_LEVENBERG_MARQUARDT_H__

#include <Eigen/Dense>
#include "solver.h"
namespace pd{namespace vision{

    template<typename Problem, typename Loss>
    class LevenbergMarquardt : public Solver<Problem>{
        using Mmxn = Eigen::Matrix<double, Eigen::Dynamic, Problem::nParameters>;
        public:
        LevenbergMarquardt(
                int nObservations,
                int maxIterations,
                double minStepSize,
                double minGradient,
                double lambdaMin = 1e-7,
                double lambdaMax = 1e7
                );

        void solve(std::shared_ptr<Problem> problem) const override;
        void solve(std::shared_ptr<Problem> problem, Eigen::VectorXd& chi2,Eigen::VectorXd &chi2pred, Eigen::VectorXd& lambda, Eigen::VectorXd& stepSize, Mmxn& x) const;
        const int& maxIterations() const { return _maxIterations;}
        private:
        const double _minGradient, _minStepSize;
        const int _maxIterations, _nObservations;
        const double _lambdaMax = 1e7, _lambdaMin = 1e-7;
        const double _Lup = 5,_Ldown = 4; ///<Scalar to multiply lambda in case linearization was good/bad
    };
}}
    #include "LevenbergMarquardt.hpp"

#endif