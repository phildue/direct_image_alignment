#ifndef VSLAM_SE3_ALIGNMENT
#define VSLAM_SE3_ALIGNMENT

#include "core/core.h"
#include "lukas_kanade/lukas_kanade.h"
#include "solver/solver.h"
namespace pd::vision{
class SE3Alignment {
        public:
        typedef std::shared_ptr<SE3Alignment> ShPtr;
        typedef std::unique_ptr<SE3Alignment> UnPtr;
        typedef std::shared_ptr<const SE3Alignment> ConstShPtr;
        typedef std::unique_ptr<const SE3Alignment> ConstUnPtr;

        SE3Alignment(double minGradient,
         vslam::solver::Solver<6>::ShPtr solver,
         vslam::solver::Loss::ShPtr loss,
         vslam::solver::Scaler::ShPtr scaler
         );

        PoseWithCovariance::UnPtr align(FrameRgbd::ConstShPtr from, FrameRgbd::ConstShPtr to) const;
        PoseWithCovariance::UnPtr align(const std::vector<FrameRgbd::ConstShPtr>& from,  FrameRgbd::ConstShPtr to) const;

   
        protected:
     
        const double _minGradient;
        const vslam::solver::Loss::ShPtr _loss;
        const vslam::solver::Scaler::ShPtr _scaler;
        const vslam::solver::Solver<6>::ShPtr _solver;
        const bool _includePrior;

};
}
#endif// VSLAM_SE3_ALIGNMENT

