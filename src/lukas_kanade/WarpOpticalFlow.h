#ifndef VSLAM_LUKAS_KANADE_OPTICAL_FLOW_H__
#define VSLAM_LUKAS_KANADE_OPTICAL_FLOW_H__
#include <memory>
#include "core/types.h"
namespace pd{namespace vision{
class WarpOpticalFlow
{
    public:
    inline constexpr static int nParameters = 2;
    WarpOpticalFlow(const Eigen::Vector2d& x);
    void updateAdditive(const Eigen::Vector2d& dx);
    void updateCompositional(const Eigen::Vector2d& dx);
    Eigen::Vector2d apply(int u, int v) const;
    Eigen::Matrix<double,2,nParameters> J(int u, int v) const;
    
    void setX(const Eigen::Vector2d& x);
    Eigen::Vector2d x() const {return _x;}
    private:
    Eigen::Matrix3d toMat(const Eigen::Vector2d& x) const ;
    Eigen::Matrix3d _w;
    Eigen::Vector2d _x;
};


}}
#endif