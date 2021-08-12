//
// Created by phil on 02.07.21.
//
#include "algorithm.h"
namespace pd{ namespace vision{ namespace algorithm{

    std::uint8_t bilinearInterpolation(const Image& mat, double x, double y)
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

        return static_cast<int>(P);
    }


    double rmse(const Eigen::MatrixXi& patch1, const Eigen::MatrixXi& patch2)
    {
        if (patch1.rows() != patch2.rows() || patch1.cols() != patch2.cols())
        {
            throw std::runtime_error("rmse:: Patches have unequal dimensions!");
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

    Image resize(const Image& mat, double scale) {
        return Image();
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


}}}