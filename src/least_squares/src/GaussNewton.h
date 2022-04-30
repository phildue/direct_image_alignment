#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__
#include <memory>

#include <Eigen/Dense>
#include <core/core.h>
#include "Solver.h"

namespace pd::vslam::least_squares{

      template<int nParameters>
      class GaussNewton : public Solver<nParameters>{
        typedef Eigen::Matrix<double, Eigen::Dynamic, nParameters> Mmxn;
        using Vn = Eigen::Matrix<double, nParameters, 1>;
 
        public:
        typedef std::shared_ptr<GaussNewton> ShPtr;
        typedef std::unique_ptr<GaussNewton> UnPtr;
        typedef std::shared_ptr<const GaussNewton> ConstShPtr;
        typedef std::unique_ptr<const GaussNewton> ConstUnPtr;

        GaussNewton( double minStepSize, size_t maxIterations );

        typename Solver<nParameters>::Results::ConstUnPtr solve(std::shared_ptr< Problem<nParameters> > problem) override;

        private:
        const double _minStepSize;
        const double _minGradient;
        const double _minReduction;
        const size_t _maxIterations;
        
    };
   
}
#include "GaussNewton.hpp"
#endif