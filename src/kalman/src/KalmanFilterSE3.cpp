#include "KalmanFilterSE3.h"

/*
* We measure the relative motion as SE3 pose between frame_t and frame t_1
* But try to estimate the velocity which we assume to be constant with some uncertainty (process noise)
* Where we use the translation vector and the angle axis rotation as representation
*
* State x       [px, py, pz, ax, ay, az, vx, vy, vz, avx, avy, avz]
* Measurement z [dpx, dpy, dpz, dax, day, daz]
*
* H = [ dt, 0, 0, 0, 0, 0,
        0, dt, 0, 0, 0, 0,
        0, 0, dt, 0, 0, 0,
        0, 0, 0, dt, 0, 0,
        0, 0, 0, 0, dt, 0,
        0, 0, 0, 0, 0, dt]

* A = [ 1, 0, 0, 0, 0, 0, dt, 0,  0,  0,  0,  0, 
        0, 1, 0, 0, 0, 0, 0, dt,  0,  0,  0,  0,
        0, 0, 1, 0, 0, 0, 0,  0, dt,  0,  0,  0,
        0, 0, 0, 1, 0, 0, 0,  0,  0, dt,  0,  0,
        0, 0, 0, 0, 1, 0, 0,  0,  0,  0, dt,  0,
        0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0, dt,
        ]
*
*
*
*
*
*/

namespace pd::vision{

        KalmanFilterSE3::KalmanFilterSE3(const Matd<6,6>& Q, const Matd<6,1>& x0, std::uint64_t t0)
        : KalmanFilter<6,6>(Matd<6,6>::Zero(),x0,t0)
        {    
                _Q = Q;          
        }

        Matd<6,6> KalmanFilterSE3::A(std::uint64_t UNUSED(dT) ) const
        {       //we assume constant velocity
                return Matd<6,6>::Identity();
        }

        Matd<6,6> KalmanFilterSE3::H(std::uint64_t dT) const
        {       //we measure relative motion so we have to scale the state with dT
                return Matd<6,6>::Identity()*dT;
        }

}