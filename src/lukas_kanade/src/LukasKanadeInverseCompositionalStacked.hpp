
#include "utils/utils.h"
#include "core/core.h"

namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositionalStacked<Warp>::LukasKanadeInverseCompositionalStacked (const std::vector<Image>& templ, const Image& image,const std::vector<std::shared_ptr<Warp>>& w0, std::shared_ptr<Loss> l, double minGradient)
    {
        if ( templ.size() != w0.size() )
        {
            throw pd::Exception(" Each template needs a warp ");
        }
        _frames.resize(templ.size());
        const size_t nFrames = _frames.size();

        std::vector<Matd<-1, Warp::nParameters>> js( nFrames );

        size_t nTotalRows = 0U;
        for ( size_t i = 0; i < nFrames; i++)
        {
            _frames[i] = std::make_shared<LukasKanadeInverseCompositional<Warp>>(templ[i], image, w0[i], l, minGradient);
            Matd<-1, Warp::nParameters> jFrame;
            _frames[i]->computeJacobian(jFrame);
            nTotalRows += jFrame.size();
            js[i] = std::move(jFrame);
        }

        _J.conservativeResize(nTotalRows,Eigen::NoChange);
        size_t idx = 0U;
        for ( size_t i = 0; i < nFrames; i++ )
        {
            _J.middleRows( idx, js[i].size() ) = js[i];
            idx += js[i].size();
        }
    }

    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w)
    {

        std::vector<VecXd> rs(_frames.size()),ws(_frames.size());
        size_t nTotalRows = 0U;

        for ( size_t i = 0; i < _frames.size(); i++ )
        {
            VecXd rFrame, wFrame;
            _frames[i]->computeResidual( rFrame, wFrame );
            nTotalRows += rFrame.size();
            rs[i] = std::move(rFrame);
            ws[i] = std::move(wFrame);

        }
        
        r.conservativeResize(nTotalRows);
        w.conservativeResize(nTotalRows);
        size_t idx = 0U;
        for ( size_t i = 0; i < _frames.size(); i++ )
        {
            r.middleRows( idx, rs[i].size() ) = rs[i];
            w.middleRows( idx, rs[i].size() ) = ws[i];
            idx += rs[i].size();
        }

    }

    //
    // J = Ixy*dW/dp
    //
    template<typename Warp>
    bool LukasKanadeInverseCompositionalStacked<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& j)
    {
        j = _J;//TODO does this do a copy?
        return true;
    }

    

    template<typename Warp>
    bool LukasKanadeInverseCompositionalStacked<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        for ( auto& f : _frames )
        {
            f->updateX(dx);
        }
        return true;
    }

    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::extendLeft(Eigen::MatrixXd& UNUSED(H))
    {}
    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::extendRight(Eigen::VectorXd& UNUSED(g))
    {}

}}