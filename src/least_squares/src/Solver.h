#ifndef VSLAM_SOLVER_H__
#define VSLAM_SOLVER_H__

namespace pd::vslam::least_squares{

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
                      A.noalias() += that.A;
                      b.noalias() += that.b;
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