#pragma once

#include "core/result.hpp"
#include "model/model.hpp"
#include <string>
#include <istream>

namespace sih26119 {

class MpsReader {
public:
    MpsReader() = default;

    Result<Model> read_file(const std::string& filepath);
    Result<Model> read_stream(std::istream& is, const std::string& sourcename = "stream");
};

} // namespace sih26119
