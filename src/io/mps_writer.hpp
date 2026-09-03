#pragma once

#include "core/status.hpp"
#include "model/model.hpp"
#include <string>
#include <ostream>

namespace sih26119 {

class MpsWriter {
public:
    MpsWriter() = default;

    Status write_file(const Model& model, const std::string& filepath);
    Status write_stream(const Model& model, std::ostream& os);
};

} // namespace sih26119
