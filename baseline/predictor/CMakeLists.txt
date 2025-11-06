project(predictor CXX)

# Automatic Dependencies
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(Eigen3)

find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)
find_package(gemmi CONFIG REQUIRED)

add_executable(baseline_predictor
    predict.cc
)
target_link_libraries(baseline_predictor
    PRIVATE
    fmt
    dx2
    argparse
    Bitshuffle::bitshuffle
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    version
    ffs_common
    Eigen3::Eigen
    gemmi::gemmi_cpp
)
