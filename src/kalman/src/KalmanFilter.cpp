#include "KalmanFilter.h"
namespace pd::vision{

        KalmanFilter::KalmanFilter(const MatD& B, const MatD& C,  const MatD& P, const VecD& x0, std::uint64_t t0)
        :_B(B)
        ,_C(C)
        ,_P(P)
        ,_t(t0)
        ,_x(x0)
        {
                _K = MatD::Zero(C.cols(),C.rows());
        }

        VecD KalmanFilter::predict(std::uint64_t t)
        {
                return A(t - _t) * _x;
        }
        void KalmanFilter::update(std::uint64_t t,  const VecD& z, const MatD& R)
        {
                const MatD Ax = predict(t);
                _x = Ax + _K * ( z - _C * Ax);

                _K.noalias() = _P * _C.transpose() * R.inverse() - _K * _C * _P * _C.transpose() * R.inverse();
                _P.noalias() = _P - _K * _C * _P;
                _t = t;
        }
}