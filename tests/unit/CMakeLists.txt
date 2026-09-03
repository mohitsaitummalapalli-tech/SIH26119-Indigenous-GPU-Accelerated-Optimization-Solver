add_executable(test_infrastructure test_infrastructure.cpp)

target_include_directories(test_infrastructure PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

add_test(NAME InfrastructureTest COMMAND test_infrastructure)
