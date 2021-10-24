#ifndef VSLAM_SOLVER_H__
#define VSLAM_SOLVER_H__

template<int nParameters>
class Solver
{
    public:
    using Vn = Eigen::Matrix<double, nParameters, 1>;
    virtual void solve(Eigen::Matrix<double, nParameters, 1>& x) const = 0;
};

#endif