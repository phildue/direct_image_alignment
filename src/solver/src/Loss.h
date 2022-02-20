#ifndef VSLAM_LOSS_H__
#define VSLAM_LOSS_H__

#include <memory>

#include <eigen3/Eigen/Dense>
#include "core/core.h"

// computeWeights assigns weights corresponding to L'(r) the first order derivative of the loss function
// There are many other weights based on the expected distribution of errors (e.g. t distribution)[Image Gradient-based Joint Direct Visual Odometry for Stereo Camera]

namespace pd{namespace vision{
class Loss
{
        public:
        typedef std::shared_ptr<Loss> ShPtr;
        typedef std::unique_ptr<Loss> UnPtr;
        typedef std::shared_ptr<const Loss> ConstShPtr;
        typedef std::unique_ptr<const Loss> ConstUnPtr;

         //l(r)
        virtual double compute(double r) const = 0;
        //dl/dr
        virtual double computeDerivative(double r) const = 0;
        //w(r) = dl/dr (r) * 1/r
        virtual double computeWeight(double r) const {return computeDerivative(r)/r;}
     
};

class QuadraticLoss : public Loss
{
        public:
         //l(r)
        double compute(double r) const override {return 0.5 * r*r;}
        //dl/dr
        double computeDerivative(double r) const override {return r;}
        //w(r) = dl/dr (r) * 1/r
        double computeWeight(double UNUSED(r)) const override {return 1.0;}
     
};

class TukeyLoss : public Loss
{
        public:
        inline constexpr static double C = 4.6851; //<constant from paper corresponding to the 95% asymptotic efficiency on the standard normal distribution
        inline constexpr static double C2_6 = C/6.0;
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
        HuberLoss(double c)
        :_c(c)
        {}
        const double _c;
        //w(r) = dl/dr (r) * 1/r
        double computeWeight(double r) const override;
        //dl/dr
        double computeDerivative(double r) const override;
        //l(r)
        double compute(double r) const override;

};

}}
#endif