# Phase 0: Infrastructure Verification
add_executable(test_infrastructure test_infrastructure.cpp)
target_include_directories(test_infrastructure PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)
target_compile_definitions(test_infrastructure PRIVATE
    SIH26119_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
add_test(NAME InfrastructureTest COMMAND test_infrastructure)

# Phase 1: Canonical Mathematical Model
add_executable(test_model test_model.cpp)
target_link_libraries(test_model PRIVATE sih26119_solver)
add_test(NAME ModelTest COMMAND test_model)

# Phase 1: Native MPS Parser & Semantics
add_executable(test_mps_parser test_mps_parser.cpp)
target_link_libraries(test_mps_parser PRIVATE sih26119_solver)
add_test(NAME MpsParserTest COMMAND test_mps_parser)

# Phase 1: Native LP Parser & Factor-of-Two Gate
add_executable(test_lp_parser test_lp_parser.cpp)
target_link_libraries(test_lp_parser PRIVATE sih26119_solver)
add_test(NAME LpParserTest COMMAND test_lp_parser)

# Phase 1: Model Serialization & Semantic Round-Trip
add_executable(test_serialization test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE sih26119_solver)
add_test(NAME SerializationTest COMMAND test_serialization)

# Phase 2: Dense Vector Numerical Abstraction
add_executable(test_dense_vector test_dense_vector.cpp)
target_link_libraries(test_dense_vector PRIVATE sih26119_solver)
add_test(NAME DenseVectorTest COMMAND test_dense_vector)

# Phase 2: Dense Matrix (Column-Major) Numerical Layer
add_executable(test_dense_matrix test_dense_matrix.cpp)
target_link_libraries(test_dense_matrix PRIVATE sih26119_solver)
add_test(NAME DenseMatrixTest COMMAND test_dense_matrix)

# Phase 2: Sparse Matrix (CSR) & SpMV Foundation
add_executable(test_sparse_matrix test_sparse_matrix.cpp)
target_link_libraries(test_sparse_matrix PRIVATE sih26119_solver)
add_test(NAME SparseMatrixTest COMMAND test_sparse_matrix)

# Phase 2: Numerics Stability, Tolerances & Deterministic Property Tests
add_executable(test_numerics test_numerics.cpp)
target_link_libraries(test_numerics PRIVATE sih26119_solver)
add_test(NAME NumericsTest COMMAND test_numerics)
