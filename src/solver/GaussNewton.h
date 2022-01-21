#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__
#include <memory>

#include <Eigen/Dense>

#include "Loss.h"
namespace pd{namespace vision{

      template<typename Problem>
      class GaussNewton {
        typedef Eigen::Matrix<double, Eigen::Dynamic, Problem::nParameters> Mmxn;
        using Vn = Eigen::Matrix<double, Problem::nParameters, 1>;
 
        public:
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

        private:
        const double _alpha;
        const double _minStepSize;
        const double _minGradient;
        const double _minReduction;
        const int _maxIterations;
        Eigen::VectorXd _chi2,_stepSize;
        Eigen::Matrix<double,Eigen::Dynamic,Problem::nParameters> _x;
        int _i;

    };
   
}}
#include "GaussNewton.hpp"
#endif