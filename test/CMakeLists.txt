cmake_minimum_required(VERSION 3.8)
project(direct_image_alignmentTest)
include(../tools/Macros.cmake)

enable_testing()
find_package(GTest MODULE REQUIRED)

if(NOT TARGET PD::direct_image_alignment)
	find_package(direct_image_alignment REQUIRED)
endif()

add_executable(direct_image_alignmentTests test_image_alignment.cpp)

target_link_libraries(direct_image_alignmentTests
    PRIVATE
    PD::direct_image_alignment
    GTest::Main
    )

add_test(NAME direct_image_alignment.UnitTests
    COMMAND direct_image_alignmentTests
    )
