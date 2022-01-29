//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "kalman/kalman.h"
#include "utils/utils.h"
#include "core/core.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

class KalmanFilter2D : public KalmanFilter
{
    ///state = [px py vx vy]
    ///measurement = [px py]
    public:
    KalmanFilter2D()
    : KalmanFilter(MatD::Zero(4,1),MatD::Zero(2,4),MatD::Identity(4,4),VecD::Zero(4),0U)
    {
        _C(0,0) = 1;
        _C(1,1) = 1;
        _x(2) = 0.1;
        _x(3) = 0.1;
    
        _P.noalias() = _P + MatD::Identity(4,4) * 10;
    }
    MatD A(std::uint64_t dt) const override
    {
        MatD M = MatD::Identity(4,4);
        M(0,2) = dt;
        M(1,3) = dt;
        return M;
    }
};

void plot(const std::vector<Eigen::Vector2d>& traj, std::string name)
{
    std::vector<double> x(traj.size()),y(traj.size());
    for (int i = 0; i < traj.size(); i++ )
    {
        x[i] = traj[i].x();
        y[i] = traj[i].y();
    }
    vis::plt::named_plot(name.c_str(),x,y);

}

TEST(KalmanFilterTest,Motion2DTest)
{
    std::shared_ptr<KalmanFilter2D> kalman = std::make_shared<KalmanFilter2D>();

    Eigen::Vector2d velTrue;
    velTrue << 1,2;
    const int dt = 1;
    Eigen::Vector2d cov;
    cov.x() = 1;
    cov.y() = 1;
    std::vector<Eigen::Vector2d> trajTrue(100);
    std::vector<Eigen::Vector2d> trajNoise(100);
    std::vector<Eigen::Vector2d> trajKalman(100);
    trajKalman[0].setZero();
    trajNoise[0].setZero();
    trajTrue[0].setZero();

    for(int t = 1; t < 100; t+=dt)
    {
        trajKalman[t] = kalman->predict(t);
        trajTrue[t] = trajTrue[t-1] + (double)dt * velTrue;
        trajNoise[t] = trajTrue[t] + random::N(cov);
        kalman->update(t, trajNoise[t], cov.asDiagonal());

        std::cout << "Kalman: " << trajKalman[t].transpose();
        std::cout << "True: " << trajKalman[t].transpose();
        std::cout << "Noise: " << trajKalman[t].transpose();

    }

    vis::plt::figure();
    plot(trajTrue,"True");
    plot(trajNoise,"Noise");
    vis::plt::legend();
    
    vis::plt::figure();
    plot(trajKalman,"Kalman");

    vis::plt::legend();
    vis::plt::show();
}


