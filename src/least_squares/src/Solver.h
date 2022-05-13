#ifndef VSLAM_SOLVER_H__
#define VSLAM_SOLVER_H__

#include "NormalEquations.h"
namespace pd::vslam::least_squares{

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
              
              virtual NormalEquations::ConstShPtr computeNormalEquations() = 0;
      };


      template<int nParameters>
      class Solver{
          public:
          typedef std::shared_ptr<Solver> ShPtr;
          typedef std::unique_ptr<Solver> UnPtr;
          typedef std::shared_ptr<const Solver> ConstShPtr;
          typedef std::unique_ptr<const Solver> ConstUnPtr;
          struct Results{
                typedef std::shared_ptr<Results> ShPtr;
                typedef std::unique_ptr<Results> UnPtr;
                typedef std::shared_ptr<const Results> ConstShPtr;
                typedef std::unique_ptr<const Results> ConstUnPtr;

                Eigen::VectorXd chi2,stepSize;
                Eigen::Matrix<double,Eigen::Dynamic,nParameters> x;
                std::vector<Eigen::Matrix<double,nParameters,nParameters>> cov;
                size_t iteration;
        };
          
          virtual typename Results::ConstUnPtr solve(std::shared_ptr< Problem<nParameters> > problem) = 0;
      };
}
#endif