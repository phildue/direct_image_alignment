#ifndef VSLAM_WARP_H__
#define VSLAM_WARP_H__
#include <memory>
#include <vector>

#include "core/core.h"
namespace pd::vslam::lukas_kanade{
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

/*
Warp based reprojection with SE3 (T) transformation:
uv_1 = p( T * p^-1( uv_0, Z(uv_0) ) )
*/
class WarpSE3
{
    public:
    inline constexpr static int nParameters = 6;
    WarpSE3(const SE3d& poseCur, const Eigen::MatrixXd& depth, Camera::ConstShPtr camRef, Camera::ConstShPtr camCur, const SE3d& poseRef = {});
    WarpSE3(const SE3d& poseCur, const std::vector<Vec3d>& pcl, size_t width, Camera::ConstShPtr camRef, Camera::ConstShPtr camCur, const SE3d& poseRef = {});
    void updateAdditive(const Eigen::Vector6d& dx);
    void updateCompositional(const Eigen::Vector6d& dx);

    Eigen::Vector2d apply(int u, int v) const ;
    Eigen::Matrix<double,2,nParameters> J(int u, int v) const;

    Image apply(const Image& img) const ;
    DepthMap apply(const DepthMap& img) const ;

    void setX(const Eigen::Vector6d& x);
    Eigen::Vector6d x() const;
    SE3d poseCur() const;

    private:
    Eigen::Vector6d _x;
    SE3d _se3,_poseRef;
    const int _width;
    const std::shared_ptr<const Camera> _camCur, _camRef;
    std::vector<Eigen::Vector3d> _pcl;
};

}
#endif