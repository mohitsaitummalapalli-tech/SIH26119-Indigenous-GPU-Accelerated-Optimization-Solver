add_library(sih26119_solver STATIC
    model/model.cpp
    io/mps_reader.cpp
    io/mps_writer.cpp
    io/lp_reader.cpp
    io/lp_writer.cpp
)

target_include_directories(sih26119_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
