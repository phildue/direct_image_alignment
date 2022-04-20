#ifndef VSLAM_SOLVER_SCALER
#define VSLAM_SOLVER_SCALER
#include <core/core.h>

namespace pd::vslam::least_squares{


class Scaler{
        public:
        typedef std::shared_ptr<Scaler> ShPtr;
        typedef std::unique_ptr<Scaler> UnPtr;
        typedef std::shared_ptr<const Scaler> ConstShPtr;
        typedef std::unique_ptr<const Scaler> ConstUnPtr;

        virtual VecXd scale(const VecXd& r) {return r;};
        virtual void compute(const VecXd& UNUSED(r)){};
        virtual double scale(double UNUSED(r)) {return 1.0;};

};

class MedianScaler : public Scaler{
        public:
        VecXd scale(const VecXd& r) override;
        double scale(double r) override { return (r - _median)/_std;}
        void compute(const VecXd& r) override;
        private:
        double _median = 0.0;
        double _std = 1.0;
};
class MeanScaler : public Scaler{
        public:
        VecXd scale(const VecXd& r) override;
        double scale(double r) override { return (r - _mean)/_std;}
        void compute(const VecXd& r) override;
        private:
        double _mean = 0.0;
        double _std = 1.0;
};

class ScalerTDistribution : public Scaler{
        public:
        ScalerTDistribution(double v = 5.0):_v(v){}
        VecXd scale(const VecXd& r) override;
        private:
        const double _v = 5.0; //Experimentally, Robust Odometry Estimation From Rgbd Cameras
        double _sigma = 1.0;
        uint64_t _maxIterations = 20;
        double _minStepSize = 1e-7;
};
} // namespace pd::vslam::least_squares

#endif