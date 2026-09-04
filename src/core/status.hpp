#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <iostream>

namespace sih26119 {

enum class StatusCode {
    Ok,
    InvalidArgument,
    DuplicateVariableName,
    DuplicateConstraintName,
    InvalidVariableReference,
    InvalidConstraintReference,
    InvalidBounds,
    MalformedQuadraticTerm,
    InvalidBinaryDeclaration,
    IOError,
    ParseError,
    UnsupportedFeature,
    InconsistentModel,
    NumericalFailure
};

inline std::string_view status_code_to_string(StatusCode code) noexcept {
    switch (code) {
        case StatusCode::Ok: return "Ok";
        case StatusCode::InvalidArgument: return "InvalidArgument";
        case StatusCode::DuplicateVariableName: return "DuplicateVariableName";
        case StatusCode::DuplicateConstraintName: return "DuplicateConstraintName";
        case StatusCode::InvalidVariableReference: return "InvalidVariableReference";
        case StatusCode::InvalidConstraintReference: return "InvalidConstraintReference";
        case StatusCode::InvalidBounds: return "InvalidBounds";
        case StatusCode::MalformedQuadraticTerm: return "MalformedQuadraticTerm";
        case StatusCode::InvalidBinaryDeclaration: return "InvalidBinaryDeclaration";
        case StatusCode::IOError: return "IOError";
        case StatusCode::ParseError: return "ParseError";
        case StatusCode::UnsupportedFeature: return "UnsupportedFeature";
        case StatusCode::InconsistentModel: return "InconsistentModel";
        case StatusCode::NumericalFailure: return "NumericalFailure";
    }
    return "UnknownStatus";
}

class Status {
public:
    Status() noexcept : code_(StatusCode::Ok), line_(0) {}

    explicit Status(StatusCode code, std::string message = "", uint64_t line = 0)
        : code_(code), message_(std::move(message)), line_(line) {}

    static Status ok() noexcept {
        return Status();
    }

    static Status error(StatusCode code, std::string message = "") {
        return Status(code, std::move(message), 0);
    }

    static Status parse_error(uint64_t line, std::string message) {
        return Status(StatusCode::ParseError, std::move(message), line);
    }

    [[nodiscard]] bool is_ok() const noexcept {
        return code_ == StatusCode::Ok;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return is_ok();
    }

    [[nodiscard]] bool operator!() const noexcept {
        return !is_ok();
    }

    [[nodiscard]] StatusCode code() const noexcept {
        return code_;
    }

    [[nodiscard]] const std::string& message() const noexcept {
        return message_;
    }

    [[nodiscard]] uint64_t line() const noexcept {
        return line_;
    }

    [[nodiscard]] std::string to_string() const {
        if (is_ok()) {
            return "Status::Ok";
        }
        std::string res = std::string(status_code_to_string(code_));
        if (line_ > 0) {
            res += " (line " + std::to_string(line_) + ")";
        }
        if (!message_.empty()) {
            res += ": " + message_;
        }
        return res;
    }

    friend std::ostream& operator<<(std::ostream& os, const Status& status) {
        os << status.to_string();
        return os;
    }

private:
    StatusCode code_;
    std::string message_;
    uint64_t line_;
};

} // namespace sih26119
