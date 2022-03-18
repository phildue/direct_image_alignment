#ifndef VSLAM_SOLVER_SCALER
#define VSLAM_SOLVER_SCALER
#include <core/core.h>

namespace pd::vslam::solver
{

class Scaler{
        public:
        typedef std::shared_ptr<Scaler> ShPtr;
        typedef std::unique_ptr<Scaler> UnPtr;
        typedef std::shared_ptr<const Scaler> ConstShPtr;
        typedef std::unique_ptr<const Scaler> ConstUnPtr;

        virtual vision::VecXd scale(const vision::VecXd& r) {return r;};

};

class MedianScaler : public Scaler{
        public:
        vision::VecXd scale(const vision::VecXd& r) override;
};

class ScalerTDistribution : public Scaler{
        public:
        ScalerTDistribution(double v):_v(v){}
        vision::VecXd scale(const vision::VecXd& r) override;
        private:
        const double _v = 5.0; //Experimentally, Robust Odometry Estimation From Rgbd Cameras
        double _sigma = 1.0;
        uint64_t _maxIterations = 20;
        double _minStepSize = 1e7;
};
} // namespace pd::vslam::solver

#endif