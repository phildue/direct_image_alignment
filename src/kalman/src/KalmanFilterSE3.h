#ifndef VSLAM_KALMAN_FILTER_SE3_H__
#define VSLAM_KALMAN_FILTER_SE3_H__

#include "core/core.h"
#include "KalmanFilter.h"
namespace pd::vision{

class KalmanFilterSE3 : public KalmanFilter<6,6>{
        public:
        typedef std::shared_ptr<KalmanFilterSE3> ShPtr;
        typedef std::unique_ptr<KalmanFilterSE3> UnPtr;
        typedef std::shared_ptr<const KalmanFilterSE3> ConstPtr;

        KalmanFilterSE3(const Matd<6,6>& Q,const Matd<6,1>& x0, std::uint64_t t0 = std::numeric_limits<uint64_t>::max());
        Matd<6,6> A(std::uint64_t dT) const override;
        Matd<6,6> H(std::uint64_t dT) const override;

};
}
#endif //VSLAM_KALMAN_H__