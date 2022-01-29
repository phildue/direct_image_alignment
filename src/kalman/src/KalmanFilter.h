#ifndef VSLAM_KALMAN_FILTER_H__
#define VSLAM_KALMAN_FILTER_H__

#include "core/core.h"
namespace pd::vision{

/// Bayesian filtering for a linear system following:
/// x_t1 = A x_t0 + B u_t0
/// With measurement model:
/// z_t = C x_t
/// Assuming gaussian noise on system
/// p(x_t) = N(x_t,P)
/// Assuming gaussian noise on measurement:
/// z_t = N(z_1,R)
class KalmanFilter{
        public:
                KalmanFilter(const MatD& B, const MatD& C, const MatD& P, const VecD& x0, std::uint64_t t0);
                VecD predict(std::uint64_t t);
                MatD cov() const {return _P;}
                void update(std::uint64_t t,  const VecD& z, const MatD& R);
                virtual MatD A(std::uint64_t dT) const = 0;
        protected:
        MatD _B; //< Control matrix (?) modeling impact of control input, maps from control space to state space
        MatD _C; //< Observation matrix encoding measurement model, maps from measurement space to state space
        MatD _P; //< System Covariance
        MatD _K; //< Kalman gain
        std::uint64_t _t; //< last update
        VecD _x; //< state


};
}
#endif //VSLAM_KALMAN_H__