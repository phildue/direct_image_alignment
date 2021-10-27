#include "LukasKanadeAffine.h"
#include "solver/LevenbergMarquardt.h"
#include "utils/visuals.h"
namespace pd{namespace vision{

    LukasKanadeAffine::LukasKanadeAffine (const Image& templ, const Image& image, int maxIterations, double minStepSize, double minGradient)
    : LukasKanade<6>(templ,image,maxIterations,minStepSize,minGradient)
    {

    }

    bool LukasKanadeAffine::updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) const
    {
        x.noalias() += dx;
        return true;
    }

  
    Eigen::MatrixXi LukasKanadeAffine::warp(const Eigen::Matrix<double,6,1>& x, const Eigen::MatrixXi& img) const
    {
        Eigen::Matrix3d warp;
        warp << 1+x(0),   x(2), x(4),
                  x(1), 1+x(3), x(5),
                     0,      0,    1;
        Eigen::MatrixXi out = img;
        algorithm::warpAffine(img,warp,out);
        return out;
    }

    Image LukasKanadeAffine::warp(const Eigen::Matrix<double,6,1>& x, const Image& img) const
    {
        Eigen::Matrix3d warp;
        warp << 1+x(0),   x(2), x(4),
                  x(1), 1+x(3), x(5),
                     0,      0,    1;
        Image out = img;
        algorithm::warpAffine(img,warp,out);
        return out;
    }

    Eigen::Matrix<double,2,6> LukasKanadeAffine::jacobianWarp(int v, int u) const
    {
        const double cx = _T.cols()/2;
        const double cy = _T.rows()/2;
    
        Eigen::Matrix<double,2,6> Jwarp;
        Jwarp << u - cx,0,v - cy,0,1,0,
                0,u - cx,0,v - cy,0,1;
        return Jwarp;
             
    }

}}