#ifndef VSLAM_WARP_H__
#define VSLAM_WARP_H__
#include <memory>
#include "core/core.h"
#include "solver/solver.h"
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

class WarpOpticalFlow
{
    public:
    inline constexpr static int nParameters = 2;
    WarpOpticalFlow(const Eigen::Vector2d& x);
    void updateAdditive(const Eigen::Vector2d& dx);
    void updateCompositional(const Eigen::Vector2d& dx);
    Eigen::Vector2d apply(int u, int v) const;
    Eigen::Matrix<double,2,nParameters> J(int UNUSED(u), int UNUSED(v)) const;
    
    void setX(const Eigen::Vector2d& x);
    Eigen::Vector2d x() const {return _x;}
    private:
    Eigen::Matrix3d toMat(const Eigen::Vector2d& x) const ;
    Eigen::Matrix3d _w;
    Eigen::Vector2d _x;
};

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
    const SE3d& pose() const;
    const DepthMap& depth() const { return _depth;}
    static WarpSE3 resize(const WarpSE3& w, double scale);
    private:
    
    Eigen::Vector6d _x;
    SE3d _pose;
    const Eigen::MatrixXd _depth;
    const std::shared_ptr<const Camera> _cam;
};

}}
#endif