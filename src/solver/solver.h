#ifndef VSLAM_SOLVER_H__
#define VSLAM_SOLVER_H__

template<typename Problem>
class Solver
{
    public:
    using Vn = Eigen::Matrix<double, Problem::nParameters, 1>;
    virtual void solve(std::shared_ptr<Problem> problem) const = 0;
};

#endif