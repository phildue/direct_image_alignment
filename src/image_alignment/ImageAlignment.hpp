#include <utility>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include "core/Camera.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"

#include "utils/utils.h"

#include "SE3Parameterization.h"
#include "ImageAlignment.h"

namespace pd{ namespace vision{

    template<int patchSize>
    void ImageAlignment<patchSize>::align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const
    {
        for (int level = _levelMax; level >= _levelMin; --level)
        {
            auto pose = (targetFrame->pose() * referenceFrame->pose().inverse()).log();
            VLOG(4) << "IA init: " << " Level: " << level  << " #Features: " << referenceFrame->features().size();

            ceres::Problem problem;
            problem.AddParameterBlock(pose.data(),Sophus::SE3d::DoF,new SE3atParam());

            for ( int idxF = 0; idxF < referenceFrame->features().size(); idxF++)
            {
                const auto& f = referenceFrame->features()[idxF];
                if ( f->point() )
                {
                    if ( referenceFrame->isVisible(f->position(),patchSize, level))
                    {
                        auto cost = new ImageAlignment<patchSize>::Cost(f,targetFrame,level);
                        problem.AddResidualBlock(cost, new ceres::HuberLoss(0.001),pose.data());

                    }
                }
            }
            VLOG(4) << "Setup IA with #Parameters: " << problem.NumParameters() << ", #Residuals: " << problem.NumResiduals();
            ceres::Solver::Options options{};
            options.linear_solver_type = ceres::DENSE_SCHUR;

            options.minimizer_progress_to_stdout = VLOG_IS_ON(3);

            ceres::Solver::Summary summary{};
            ceres::Solve(options, &problem, &summary);
            targetFrame->setPose(Sophus::SE3d::exp(pose)*referenceFrame->pose());

            Log::logReprojection(referenceFrame,targetFrame,patchSize/2,4);

            if (VLOG_IS_ON(4))
            {
                VLOG(4) << summary.FullReport();

            }else{
                VLOG(3) << summary.BriefReport();

            }
        }

    }


    template<int patchSize>
    ImageAlignment<patchSize>::ImageAlignment(uint32_t levelMax, uint32_t levelMin)
    : _levelMax(levelMax)
    , _levelMin(levelMin)
    {}

    }
}
