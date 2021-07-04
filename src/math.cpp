//
// Created by phil on 02.07.21.
//
#include "math.h"
namespace pd{
    namespace vision{
        namespace math{

            double bilinearInterpolation(const Eigen::MatrixXd& mat, double x, double y)
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
                const int x1 = static_cast<int> ( std::floor(x) );
                const int x2 = static_cast<int> ( std::ceil(x));
                const int y1 = static_cast<int> ( std::floor(y) );
                const int y2 = static_cast<int> ( std::ceil(y));
                const double& Q11 = mat(x1,y1);
                const double& Q12 = mat(x1,y2);
                const double& Q21 = mat(x2,y1);
                const double& Q22 = mat(x2,y2);
                const double R1 = ((x2 - x)/(x2 - x1))*Q11 + ((x - x1)/(x2 - x1))*Q21;

                const double R2 = ((x2 - x)/(x2 - x1))*Q12 + ((x - x1)/(x2 - x1))*Q22;

                //After the two R values are calculated, the value of P can finally be calculated by a weighted average of R1 and R2.

                const double P = ((y2 - y)/(y2 - y1))*R1 + ((y - y1)/(y2- y1))*R2;

                return P;
            }


            double rmse(const Eigen::MatrixXd& patch1, const Eigen::MatrixXd& patch2)
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

            Eigen::MatrixXd resize(const Eigen::MatrixXd &mat, double scale) {
                return Eigen::MatrixXd();
            }

        }
    }
}