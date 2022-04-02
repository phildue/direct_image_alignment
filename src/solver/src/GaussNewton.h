#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__
#include <memory>

#include <Eigen/Dense>
#include <core/core.h>
namespace pd::vslam::solver{

      template<int nParameters>
      class NormalEquations{
              public:
              typedef std::shared_ptr<NormalEquations> ShPtr;
              typedef std::unique_ptr<NormalEquations> UnPtr;
              typedef std::shared_ptr<const NormalEquations> ConstShPtr;
              typedef std::unique_ptr<const NormalEquations> ConstUnPtr;
            
              Eigen::Matrix<double,nParameters,nParameters> A = Eigen::Matrix<double,nParameters,nParameters>::Zero();
              Eigen::Vector<double,nParameters> b = Eigen::Vector<double,nParameters>::Zero();
              double chi2 = 0.0;
              size_t nConstraints = 0U;
              
              void addConstraint(const Eigen::Vector<double,nParameters>& J, double r, double w)
              {
                      A.noalias() += J*J.transpose()*w;
                      b.noalias() += J*r*w;
                      chi2 += r*r*w;
                      nConstraints++;
              }
              void combine(const NormalEquations& that)
              {
                      A += that.A;
                      b += that.b;
                      chi2 += that.chi2;
                      nConstraints += that.nConstraints;
              }
              //TODO operators+
      };


      template<int nParameters>
      class Problem{
              public:
              typedef std::shared_ptr<Problem> ShPtr;
              typedef std::unique_ptr<Problem> UnPtr;
              typedef std::shared_ptr<const Problem> ConstShPtr;
              typedef std::unique_ptr<const Problem> ConstUnPtr;
            
              virtual void updateX(const Eigen::Vector<double,nParameters>& dx) = 0;
              virtual void setX(const Eigen::Vector<double,nParameters>& x) = 0;
              virtual Eigen::Vector<double,nParameters> x() const = 0;
              virtual typename NormalEquations<nParameters>::ConstShPtr computeNormalEquations() = 0;
      };


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
        GaussNewton(
                double minStepSize,
                int maxIterations
                ):GaussNewton(0.0,minStepSize,maxIterations){}
        void solve(std::shared_ptr< Problem<nParameters> > problem) override;
        const int& iteration() const {return _i;}
        const Eigen::VectorXd& chi2() const {return _chi2;}
        const Eigen::Matrix<double,Eigen::Dynamic,nParameters>& x() const {return _x;}
        const Eigen::VectorXd& stepSize() const {return _stepSize;}

        // Source: https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
        vision::Mat<double,nParameters,nParameters> cov() const override{ return _H.inverse();}
        vision::Mat<double,nParameters,nParameters> covScaled() const { return _H.inverse() * _chi2(_i)/(_i - nParameters);}

        private:
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