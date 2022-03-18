#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__
#include <memory>

#include <Eigen/Dense>
#include <core/core.h>
#include "Loss.h"
#include "Problem.h"
namespace pd::vslam::solver{

      template<int nParameters>
      class Solver{
          public:
          typedef std::shared_ptr<Solver> ShPtr;
          typedef std::unique_ptr<Solver> UnPtr;
          typedef std::shared_ptr<const Solver> ConstShPtr;
          typedef std::unique_ptr<const Solver> ConstUnPtr;
          
          virtual void solve(std::shared_ptr< Problem<nParameters> > problem) = 0;
          virtual vision::Mat<double,nParameters,nParameters> cov() const  = 0;

      };

      template<int nParameters>
      class GaussNewton : public Solver<nParameters>{
        typedef Eigen::Matrix<double, Eigen::Dynamic, nParameters> Mmxn;
        using Vn = Eigen::Matrix<double, nParameters, 1>;
 
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

        void solve(std::shared_ptr< Problem<nParameters> > problem) override;
        const int& iteration() const {return _i;}
        const Eigen::VectorXd& chi2() const {return _chi2;}
        const Eigen::Matrix<double,Eigen::Dynamic,nParameters>& x() const {return _x;}
        const Eigen::VectorXd& stepSize() const {return _stepSize;}

        // Source: https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
        vision::Mat<double,nParameters,nParameters> cov() const override{ return _H.inverse();}
        vision::Mat<double,nParameters,nParameters> covScaled() const { return _H.inverse() * _chi2(_i)/(_i - nParameters);}

        private:
        const double _alpha;
        const double _minStepSize;
        const double _minGradient;
        const double _minReduction;
        const int _maxIterations;
        Eigen::VectorXd _chi2,_stepSize;
        Eigen::Matrix<double,Eigen::Dynamic,nParameters> _x;
        int _i;
        vision::Mat<double,nParameters,nParameters> _H;

    };
   
}
#include "GaussNewton.hpp"
#endif