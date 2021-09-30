//
// Created by phil on 02.07.21.
//
#include "utils/Exceptions.h"
#include "algorithm.h"
namespace pd{ namespace vision{ namespace algorithm{



    std::uint8_t bilinearInterpolation(const Image& mat, double x, double y)
    {
        return bilinearInterpolation<std::uint8_t>(mat,x,y);
    }

    double bilinearInterpolation(const Eigen::MatrixXd& mat, double x, double y)
    {
        return bilinearInterpolation<double>(mat,x,y);

    }


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
        Eigen::MatrixXi ix = image.cast<int>().rightCols(image.cols()-1) - image.cast<int>().leftCols(image.cols()-1);
        ix.conservativeResize(Eigen::NoChange,image.cols());
        for (int i = 0; i < ix.rows() ; ++i) {
            ix.row(i).tail(1).setConstant(0);
        }
        return ix;
    }

    Eigen::MatrixXi gradY(const Image& image)
    {
        Eigen::MatrixXi iy = image.cast<int>().bottomRows(image.rows()-1) - image.cast<int>().topRows(image.rows()-1);
        iy.conservativeResize(image.rows(),Eigen::NoChange);
        for (int i = 0; i < iy.cols() ; ++i) {
            iy.col(i).tail(1).setConstant(0);
        }
        return iy;

    }

    Sophus::SE3d computeRelativeTransform(const Sophus::SE3d& t0, const Sophus::SE3d& t1)
    {
        return t1 * t0.inverse();
    }

}}}