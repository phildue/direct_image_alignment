#include "MyAlgorithm.h"
#include <iostream>
namespace pd{
namespace MyLibrary{

int MyAlgorithm::compute(int i) const
{
std::cout << "hello, I was changed" << std::endl;
	return i+1;
}


}
}
