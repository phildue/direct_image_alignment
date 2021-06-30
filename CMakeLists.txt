cmake_minimum_required(VERSION 3.8)
project(MyLibrary VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools)
include(Macros)

set(sources
		src/MyAlgorithm/MyAlgorithm.cpp
		)

set(headers
		src/MyAlgorithm/MyAlgorithm.h
		include/MyLibrary/MyLibrary.h
		)

setup_package(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" PD)


# Dependencies

# Add tests
option(MyLibrary_BUILD_TESTS "Enable to build tests" ON)
if(MyLibrary_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
