#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__

#include <Eigen/Dense>
#include "solver.h"

namespace pd{namespace vision{

      template<typename Problem, typename Loss>
      class GaussNewton : public Solver<Problem>{
        typedef Eigen::Matrix<double, Eigen::Dynamic, Problem::nParameters> Mmxn;
 
        public:
        GaussNewton(
                int nObservations,
                double alpha,
                double minStepSize,
                int maxIterations
                );

        void solve(std::shared_ptr< Problem> problem) const override;
        void solve(std::shared_ptr< Problem> problem, Eigen::VectorXd &chi2, Eigen::VectorXd& stepSize, Mmxn & x) const;
        const int& maxIterations() const {return _maxIterations;}
        private:
        const double _minStepSize, _alpha;
        const int _maxIterations, _nObservations;
    };
   
}}
#include "GaussNewton.hpp"
#endif