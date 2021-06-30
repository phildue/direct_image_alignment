from conans import ConanFile, CMake, tools


class RecipeConan(ConanFile):
    name = "direct_image_alignment"
    version = "1.0.0"
    license = "https://github.com/phildue/direct_image_alignment/blob/master/LICENSE"
    author = "<Put your name here> <And your email here>"
    url = "https://github.com/phildue/direct_image_alignment"
    description = "https://github.com/phildue/direct_image_alignment/blob/master/README.md"
    topics = ("C++", "Computer Vision", "Visual Odometry")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False], "build_tests":[True, False]}
    default_options = {"shared": False, "fPIC": True, "build_tests": False}
    generators = "cmake_paths","cmake"
    def requirements(self):
        if self.options.build_tests:
           self.requires("gtest/cci.20210126")
        self.requires("eigen/3.3.7@conan/stable")
        self.requires("sophus/1.0.0")

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def export_sources(self):
        self.copy("src/*")
        self.copy("include/*")
        self.copy("tools/*")
        self.copy("CMakeLists.txt")
        self.copy("test/*")
        
    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["CMAKE_INSTALL_PREFIX"]=self.package_folder
        cmake.definitions["MyLibrary_BUILD_TESTS"]=self.options.build_tests
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.test()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()


    def package_info(self):
        self.cpp_info.libs = [self.name]

