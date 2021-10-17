//
// Created by phil on 02.07.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H
#define DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H

#include <Eigen/Dense>
#include <algorithm>

#include "types.h"

namespace pd{ namespace vision{ namespace algorithm{


    template<typename Derived>
    double bilinearInterpolation(const Eigen::MatrixBase<Derived>& mat, double x, double y)
    {
        /*We want to interpolate P
         * http://supercomputingblog.com/graphics/coding-bilinear-interpolation/
         *
         * y2 |Q12    R2          Q22
         *    |
         * y  |       P
         *    |
         * y1 |Q11    R1          Q21
         *    _______________________
         *    x1      x            x2
         * */
        const double x1 =  std::floor(x);
        const double x2 =  std::ceil(x);
        const double y1 =  std::floor(y);
        const double y2 =  std::ceil(y);
        const double Q11 = mat(static_cast<int>(y1),static_cast<int>(x1));
        const double Q12 = mat(static_cast<int>(y1),static_cast<int>(x2));
        const double Q21 = mat(static_cast<int>(y2),static_cast<int>(x1));
        const double Q22 = mat(static_cast<int>(y2),static_cast<int>(x2));
        double R1 = 0, R2 = 0;

        if (x2 == x1)
        {
            R1 = Q11;
            R2 = Q12;
        }else{
            R1 = ((x2 - x)/(x2 - x1))*Q11 + ((x - x1)/(x2 - x1))*Q21;
            R2 = ((x2 - x)/(x2 - x1))*Q12 + ((x - x1)/(x2 - x1))*Q22;
        }

        double P = 0;
        if (y2 == y1)
        {
            P = R1;
        }else{
            //After the two R values are calculated, the value of P can finally be calculated by a weighted average of R1 and R2.
            P = ((y2 - y)/(y2 - y1))*R1 + ((y - y1)/(y2- y1))*R2;

        }

        return P;
    }
    template<typename Derived>
    double bilinearInterpolation(const Eigen::MatrixBase<Derived>& mat, const Eigen::Vector2d& xy)
    {
        return bilinearInterpolation<Derived>(mat,xy.x(),xy.y());
    }
    template< class T, typename Derived>
    Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> resize(const Eigen::MatrixBase<Derived>& mat, double scale) {
        const double scaleInv = 1.0/scale;
        Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> res( static_cast<int>(mat.rows()*scale), static_cast<int>(mat.cols()*scale));
        for (int i = 0; i < res.rows(); i++)
        {
            for (int j = 0; j < res.cols(); j++)
            {
                res(i,j) = bilinearInterpolation(mat,j*scaleInv,i*scaleInv);
            }
        }
        return res;
    }

    template <typename Derived, typename Derived1>
    void warpAffine(const Eigen::MatrixBase<Derived>& img, const Eigen::MatrixBase<Derived1>& warp, Eigen::MatrixBase<Derived>& imgWarped)
    {
        imgWarped.setZero();

        for (int v = 0; v < imgWarped.rows(); v++)
        {
            for (int u = 0; u < imgWarped.cols(); u++)
            {
                const Eigen::Vector3d uv1(u,v,1);
                const auto uvRef = warp.inverse() * uv1;
                if (1 < uvRef.x() && uvRef.x() < img.cols() - 1 &&
                    1 < uvRef.y() && uvRef.y() < img.rows() - 1 )
                {
                    imgWarped(v,u) = algorithm::bilinearInterpolation(img,uvRef.x(),uvRef.y());
                }

            }
        }
    }

    double median( const Eigen::VectorXd& d );

    double rmse(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2);
    double sad(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2);
    Image resize(const Image& mat,double scale);
    Eigen::MatrixXd resize(const Eigen::MatrixXd& mat,double scale);

    /// Computes pixel wise gradient magnitude:  sqrt( (dI/dx)^2 + (dI/dy)^2
    /// \param image grayscale image
    /// \return gradient image
    Image gradient(const Image& image);
    Eigen::MatrixXi gradX(const Image& image);
    Eigen::MatrixXi gradY(const Image& image);
    Eigen::MatrixXd normalize(const Eigen::MatrixXd& mat);
    Eigen::MatrixXd normalize(const Eigen::MatrixXd& mat,double min, double max);



    /// Computes T01 from T0 and T1
    Sophus::SE3d computeRelativeTransform(const Sophus::SE3d& t0, const Sophus::SE3d& t1);

}}}
#endif //DIRECT_IMAGE_ALIGNMENT_ALGORITHM_H
