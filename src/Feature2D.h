//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
#define DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H

#include <memory>
namespace pd{
    namespace vision{

class Feature2D {
public:
    using ShPtr = std::shared_ptr<Feature2D>;
    using ShConstPtr = std::shared_ptr<Feature2D>;

    const ShConstPtr& point() const {return _point;}
    ShConstPtr& point() {return _point;}
private:
    ShConstPtr _point;
};

    }
}

#endif //DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
