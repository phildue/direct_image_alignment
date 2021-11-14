#ifndef VSLAM_LUKAS_KANADE_AFFINE_H__
#define VSLAM_LUKAS_KANADE_AFFINE_H__
#include <memory>
#include "core/types.h"
namespace pd{namespace vision{

class WarpAffine
{
    public:
    inline constexpr static int nParameters = 6;
    WarpAffine(const Eigen::Vector6d& x, double cx,double cy);
    void updateAdditive(const Eigen::Vector6d& dx);
    void updateCompositional(const Eigen::Vector6d& dx);
    Eigen::Vector2d apply(int u, int v) const;
    Eigen::Matrix<double,2,6> J(int u, int v) const;
    void setX(const Eigen::Vector6d& x);
    Eigen::Vector6d x() const;
    private:
    Eigen::Matrix3d toMat(const Eigen::Vector6d& x) const ;
    Eigen::Matrix3d _w;
    Eigen::Vector6d _x;
    const double _cx,_cy;
};

}}
#endif