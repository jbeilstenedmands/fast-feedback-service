project(indexer CXX)

set(CMAKE_CXX_STANDARD 20)

find_package (Threads REQUIRED)

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
FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan
    GIT_TAG mdspan-0.6.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(mdspan)
FetchContent_MakeAvailable(Eigen3)


file(GLOB PREDICTOR_SOURCES *.cpp)

add_library(predictor STATIC ${PREDICTOR_SOURCES})

target_include_directories(predictor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(predictor
    PUBLIC
    dx2
    Eigen3::Eigen
    fmt
    spdlog::spdlog
)