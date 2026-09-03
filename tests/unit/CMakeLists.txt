add_executable(test_infrastructure test_infrastructure.cpp)

target_include_directories(test_infrastructure PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_compile_definitions(test_infrastructure PRIVATE
    SIH26119_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_test(NAME InfrastructureTest COMMAND test_infrastructure)
