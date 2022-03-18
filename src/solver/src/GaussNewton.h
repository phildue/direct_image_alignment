#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__
#include <memory>

#include <Eigen/Dense>
#include <core/core.h>
#include "Loss.h"
namespace pd::vslam::solver{

      template<typename Problem>
      class GaussNewton {
        typedef Eigen::Matrix<double, Eigen::Dynamic, Problem::nParameters> Mmxn;
        using Vn = Eigen::Matrix<double, Problem::nParameters, 1>;
 
        public:
        typedef std::shared_ptr<GaussNewton> ShPtr;
        typedef std::unique_ptr<GaussNewton> UnPtr;
        typedef std::shared_ptr<const GaussNewton> ConstShPtr;
        typedef std::unique_ptr<const GaussNewton> ConstUnPtr;

        GaussNewton(
                double alpha,
                double minStepSize,
                int maxIterations
                );

        void solve(std::shared_ptr< Problem> problem);
        const int& iteration() const {return _i;}
        const Eigen::VectorXd& chi2() const {return _chi2;}
        const Eigen::Matrix<double,Eigen::Dynamic,Problem::nParameters>& x() const {return _x;}
        const Eigen::VectorXd& stepSize() const {return _stepSize;}

        // Source: https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
        vision::MatXd cov() const { return _H.inverse();}
        vision::MatXd covScaled() const { return _H.inverse() * _chi2(_i)/(_i - Problem::nParameters);}

        private:
        const double _alpha;
        const double _minStepSize;
        const double _minGradient;
        const double _minReduction;
        const int _maxIterations;
        Eigen::VectorXd _chi2,_stepSize;
        Eigen::Matrix<double,Eigen::Dynamic,Problem::nParameters> _x;
        int _i;
        vision::MatXd _H;

    };
   
}
#include "GaussNewton.hpp"
#endif