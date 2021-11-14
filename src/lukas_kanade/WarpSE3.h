#ifndef VSLAM_LUKAS_KANADE_SE3_H__
#define VSLAM_LUKAS_KANADE_SE3_H__
#include <memory>
#include "core/types.h"
#include "core/Camera.h"
namespace pd{namespace vision{
class WarpSE3
{
    public:
    inline constexpr static int nParameters = 6;
    WarpSE3(const Eigen::Vector6d& x, const Eigen::MatrixXd& depth, std::shared_ptr<Camera> cam);
    void updateAdditive(const Eigen::Vector6d& dx);
    void updateCompositional(const Eigen::Vector6d& dx);
    Eigen::Vector2d apply(int u, int v) const ;
    Eigen::Matrix<double,2,nParameters> J(int u, int v) const;
    void setX(const Eigen::Vector6d& x);
    Eigen::Vector6d x() const;
    private:
    
    SE3d _pose;
    Eigen::Vector6d _x;
    const Eigen::MatrixXd _depth;
    const std::shared_ptr<const Camera> _cam;
};

}}
#endif