//
// Created by phil on 02.07.21.
//
#include "utils/Exceptions.h"
#include "algorithm.h"
namespace pd{ namespace vision{ namespace algorithm{



    double rmse(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2)
    {
        if (patch1.rows() != patch2.rows() || patch1.cols() != patch2.cols())
        {
            throw pd::Exception("rmse:: Patches have unequal dimensions!");
        }

        double sum = 0.0;
        for (int i = 0; i < patch1.rows(); i++)
        {
            for(int j = 0; j < patch2.cols(); j++)
            {
                sum += std::pow(patch1(i,j) - patch2(i,j),2);
            }
        }
        return std::sqrt( sum / (patch1.rows() * patch1.cols()));
    }

    double sad(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2)
    {

        if (patch1.rows() != patch2.rows() || patch1.cols() != patch2.cols())
        {
            throw pd::Exception("sad:: Patches have unequal dimensions!");
        }

        double sum = 0.0;
        for (int i = 0; i < patch1.rows(); i++)
        {
            for(int j = 0; j < patch2.cols(); j++)
            {
                sum += std::abs(patch1(i,j) - patch2(i,j));
            }
        }
        return sum;


    }


    Image resize(const Image& mat, double scale) {
        return resize<std::uint8_t >(mat,scale);
    }

    Eigen::MatrixXd resize(const Eigen::MatrixXd& mat, double scale) {
        return resize<double >(mat,scale);
    }

    Image gradient(const Image &image) {

        const auto ix = gradX(image);
        const auto iy = gradY(image);
        const auto grad = ix.array().pow(2) + iy.array().pow(2);
        return grad.array().sqrt().cast<std::uint8_t>();
    }

    Eigen::MatrixXi gradX(const Image& image)
    {
        Eigen::MatrixXi ix = Eigen::MatrixXi::Zero(image.rows(),image.cols());
        for (int i = 0; i < image.rows(); i++)
        {
            for(int j = 1; j < image.cols()-1; j++)
            {
                ix(i,j) = (int)((-(double)image(i,j - 1) + (double)image(i,j + 1))/2.0);
            }
        }
        return ix;
    }

    Eigen::MatrixXi gradY(const Image& image)
    {
        Eigen::MatrixXi iy = Eigen::MatrixXi::Zero(image.rows(),image.cols());
        for (int i = 1; i < image.rows()-1; i++)
        {
            for(int j = 0; j < image.cols(); j++)
            {
                iy(i,j) = (int) ((-(double)image(i-1,j) + (double)image(i+1,j))/2.0);
            }
        }
        return iy;
    }

    Sophus::SE3d computeRelativeTransform(const Sophus::SE3d& t0, const Sophus::SE3d& t1)
    {
        return t1 * t0.inverse();
    }
    Eigen::MatrixXd normalize(const Eigen::MatrixXd& mat)
    {   
        return normalize(mat, mat.minCoeff(),mat.maxCoeff());
    }
    Eigen::MatrixXd normalize(const Eigen::MatrixXd& mat,double min, double max)
    {   Eigen::MatrixXd matImage = mat;
        matImage.array() -= min;
        matImage /= (max - min);
        return matImage;
    }

    double median(const Eigen::VectorXd& d)
    {
        std::vector<double> r (d.rows());
        for ( int i = 0; i < d.rows(); i++)
        {
            r[i]=d(i);
        }
        std::sort( r.begin(), r.end() );
        const int n = r.size();
        if (n % 2 == 0)
        {
            return (r[n/2-1] + r[n/2+1])/2;
        }else{
            return r[n/2];
        }
    }
}
namespace transforms{
       Eigen::MatrixXd createdTransformMatrix2D(double x, double y, double angle) {
       Eigen::Rotation2Dd rot(angle);
       Eigen::Matrix2d r = rot.toRotationMatrix();
       Eigen::Matrix3d m;
       m << r(0,0), r(0,1),x,
            r(1,0), r(1,1),y,
                 0,      0,1;
       return m;
    }

}

namespace random{
    double U(double min, double max){
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<double> distr(min, max);
        return distr(eng);
    }
    int sign()
    {
        return U(-1,1) > 0 ? 1 : -1;
    }

}

}}