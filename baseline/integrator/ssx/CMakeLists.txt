add_subdirectory(tests)
nanobind_add_module(integrate
    integrate_module.cpp
    ${CMAKE_SOURCE_DIR}/src/integrator/sigma_estimation.cc)
