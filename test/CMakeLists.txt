cmake_minimum_required(VERSION 3.8)
project(MyLibrary)
include(../tools/Macros.cmake)

enable_testing()
find_package(GTest MODULE REQUIRED)

if(NOT TARGET PD::MyLibrary)
	find_package(MyLibrary REQUIRED)
endif()

add_executable(MyLibraryTests test_myalgorithm.cpp)

target_link_libraries(MyLibraryTests
    PRIVATE
    PD::MyLibrary
    GTest::Main
    )

add_test(NAME MyLibrary.UnitTests
    COMMAND MyLibraryTests
    )
