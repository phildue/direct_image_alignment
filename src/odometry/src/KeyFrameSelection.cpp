
#include "KeyFrameSelection.h"
namespace pd::vision{

        KeyFrameSelection::ShPtr KeyFrameSelection::make() {
                return std::make_shared<KeyFrameSelectionIdx>();
        }
}