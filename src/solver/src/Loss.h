#ifndef VSLAM_LOSS_H__
#define VSLAM_LOSS_H__

#include <memory>

#include <eigen3/Eigen/Dense>
#include "core/core.h"
#include "Scaler.h"
// computeWeights assigns weights corresponding to L'(r) the first order derivative of the loss function
// There are many other weights based on the expected distribution of errors (e.g. t distribution)[Image Gradient-based Joint Direct Visual Odometry for Stereo Camera]

namespace pd::vslam::solver{
class Loss
{
        public:
        typedef std::shared_ptr<Loss> ShPtr;
        typedef std::unique_ptr<Loss> UnPtr;
        typedef std::shared_ptr<const Loss> ConstShPtr;
        typedef std::unique_ptr<const Loss> ConstUnPtr;

        Loss(Scaler::ShPtr scaler = std::make_shared<Scaler>()):_scaler(scaler){}
        //l(r)
        virtual double compute(double r) const = 0;
        //dl/dr
        virtual double computeDerivative(double r) const = 0;
        //w(r) = dl/dr (r) * 1/r
        virtual double computeWeight(double r) const {return computeDerivative(scale(r))/scale(r);}

        virtual void computeScale(const vision::VecXd& residuals){_scaler->compute(residuals);}
        double scale(double r) const {return _scaler->scale(r);}
        private:
        Scaler::ShPtr _scaler;
     
};

class QuadraticLoss : public Loss
{
        public:
        QuadraticLoss(Scaler::ShPtr scaler = std::make_shared<Scaler>()):Loss(scaler){}

         //l(r)
        double compute(double r) const override {return 0.5 * scale(r)*scale(r);}
        //dl/dr
        double computeDerivative(double r) const override {return scale(r);}
        //w(r) = dl/dr (r) * 1/r
        double computeWeight(double UNUSED(r)) const override {return 1.0;}
     
};

class OpenCvLoss : public Loss
{
        public:
        OpenCvLoss(Scaler::ShPtr scaler = std::make_shared<Scaler>()):Loss(scaler){}

         //l(r)
        double compute(double r) const override {return 0.5 * scale(r)*scale(r);}
        //dl/dr
        double computeDerivative(double r) const override {return scale(r);}
        //w(r) = dl/dr (r) * 1/r
        double computeWeight(double r) const override {return 1.0/(_std + r);}
        void computeScale(const vision::VecXd& residuals){_std = std::sqrt((residuals.array() - residuals.mean()).array().abs().sum()/(residuals.rows() - 1));}

        private:
        double _std;
     
};

class TukeyLoss : public Loss
{
        public:
        inline constexpr static double C = 4.6851; //<constant from paper corresponding to the 95% asymptotic efficiency on the standard normal distribution
        inline constexpr static double C2_6 = C/6.0;

        TukeyLoss(Scaler::ShPtr scaler = std::make_shared<MedianScaler>()):Loss(scaler){}

        //w(r) = dl/dr (r) * 1/r
        double computeWeight(double r) const override;
        //dl/dr
        double computeDerivative(double r) const override;
        //l(r)
        double compute(double r) const override;
};        

class HuberLoss : public Loss
{
        public:
        HuberLoss(Scaler::ShPtr scaler = std::make_shared<MedianScaler>(),double c = 1.345f)
        :Loss(scaler),
        _c(c)
        {}
        const double _c;
        //w(r) = dl/dr (r) * 1/r
        double computeWeight(double r) const override;
        //dl/dr
        double computeDerivative(double r) const override;
        //l(r)
        double compute(double r) const override;

};

class LossTDistribution : public Loss
{
        public:
        LossTDistribution( Scaler::ShPtr scaler = std::make_shared<MedianScaler>(),double v = 5.0):Loss(scaler),_v(v){}
        //w(r) = dl/dr (r) * 1/r
        double computeWeight(double r) const override;
        //dl/dr
        double computeDerivative(double r) const override;
        //l(r)
        double compute(double r) const override;
        private:
        const double _v;
};

}
#endif