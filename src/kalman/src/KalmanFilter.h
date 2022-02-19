#ifndef VSLAM_KALMAN_FILTER_H__
#define VSLAM_KALMAN_FILTER_H__
//https://thekalmanfilter.com/kalman-filter-explained-simply/
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
template<int N, int M>
class KalmanFilter{
        public:
                struct Prediction{
                        Matd<N,1> state;
                        Matd<N,N> cov;
                };
                KalmanFilter(const Matd<M,N>& H, const Matd<N,N>& P0, const Matd<N,1>& x0, std::uint64_t t0)
                :_H(H)
                ,_P(P0)
                ,_K()
                ,_t(t0)
                ,_x(x0)
                ,_Q()
                {}
                Prediction predict(std::uint64_t t)
                {
                        const auto At = A(t - _t);
                        return {At * _x, At*_P*At.transpose() + _Q };
                }
                
                void update(std::uint64_t t,  const Matd<M,1>& z, const Matd<M,M>& R)
                {
                        const auto pred = predict(t);

                        _K.noalias() = pred.cov * _H.transpose() * (_H*pred.cov*_H.transpose() + R).inverse();

                        _x = pred.state + _K * ( z - _H * pred.state);
                        _P.noalias() = pred.cov - _K * _H * pred.cov;
                        _t = t;
                }

                virtual Matd<N,N> A(std::uint64_t dT) const = 0;
        protected:
        //MatD _B; //< Control matrix (?) modeling impact of control input, maps from control space to state space
        Matd<M,N> _H; //< State to measurement matrix
        Matd<N,N> _P; //< State Covariance n x n
        Matd<N,M> _K; //< Kalman gain
        std::uint64_t _t; //< last update
        Matd<N,1> _x; //< state n x 1
        Matd<N,N> _Q; //< process noise


};
}
#endif //VSLAM_KALMAN_H__