#include "LukasKanadeAffine.h"
#include "solver/LevenbergMarquardt.h"
#include "utils/visuals.h"
namespace pd{namespace vision{

    LukasKanadeAffine::LukasKanadeAffine (const Image& templ, const Image& image, int maxIterations, double minStepSize, double minGradient)
    : LukasKanade<6>(templ,image,maxIterations,minStepSize,minGradient)
    , _cx(_T.cols()/2)
    , _cy(_T.rows()/2)
    {
      
    }

     void LukasKanadeAffine::solve(Eigen::Vector6d& x) const
    {
        return _solver->solve(x);
    }
    //
    // J = Ixy*dW/dp
    //
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
        Eigen::MatrixXi out = Eigen::MatrixXi::Zero(img.rows(),img.cols());
        algorithm::warpAffine(img,warp,out);
        return out;
    }
    void LukasKanadeAffine::warp(const Eigen::Matrix<double,6,1>& x, const Image& img, Image& out, Eigen::MatrixXd& weights) const
    {
        Eigen::Matrix3d warp;
        warp << 1+x(0),   x(2), x(4),
                  x(1), 1+x(3), x(5),
                     0,      0,    1;
        out = Image::Zero(img.rows(),img.cols());
        weights = Eigen::MatrixXd::Zero(img.rows(),img.cols());
        for (int v = 0; v < out.rows(); v++)
        {
            for (int u = 0; u < out.cols(); u++)
            {
                const Eigen::Vector3d xy1(u - _cx,v - _cy,1);
                const Eigen::Vector3d xy1Ref = warp.inverse() * xy1;
                Eigen::Vector2d uvRef;
                uvRef << xy1Ref.x() + _cx,xy1Ref.y() + _cy;
                if (1 < uvRef.x() && uvRef.x() < img.cols() - 1 &&
                    1 < uvRef.y() && uvRef.y() < img.rows() - 1 )
                {
                    out(v,u) = algorithm::bilinearInterpolation(img,uvRef.x(),uvRef.y());
                    weights(v,u) = 1.0;
                }

            }
        }        
    }

    Eigen::Matrix<double,2,6> LukasKanadeAffine::jacobianWarp(int v, int u) const
    {
        Eigen::Matrix<double,2,6> Jwarp;
            Jwarp << u - _cx,0,v - _cy,0,1,0,
                     0,u - _cx,0,v - _cy,0,1;
        return Jwarp;
    }

}}