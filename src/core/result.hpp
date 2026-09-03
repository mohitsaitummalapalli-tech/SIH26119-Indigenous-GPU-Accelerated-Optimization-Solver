#pragma once

#include "core/status.hpp"
#include <variant>
#include <stdexcept>
#include <utility>

namespace sih26119 {

template <typename T>
class Result {
public:
    // Construct with successful value
    Result(const T& value) : storage_(value) {}
    Result(T&& value) : storage_(std::move(value)) {}

    // Construct with error status
    Result(Status status) : storage_(std::move(status)) {
        if (std::get<Status>(storage_).is_ok()) {
            throw std::logic_error("Result cannot be initialized with a successful Status");
        }
    }

    [[nodiscard]] bool ok() const noexcept {
        return std::holds_alternative<T>(storage_);
    }

    [[nodiscard]] bool is_ok() const noexcept {
        return ok();
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return ok();
    }

    [[nodiscard]] bool operator!() const noexcept {
        return !ok();
    }

    [[nodiscard]] const T& value() const & {
        if (!ok()) {
            throw std::runtime_error("Attempted to access value on error Result: " + status().to_string());
        }
        return std::get<T>(storage_);
    }

    [[nodiscard]] T& value() & {
        if (!ok()) {
            throw std::runtime_error("Attempted to access value on error Result: " + status().to_string());
        }
        return std::get<T>(storage_);
    }

    [[nodiscard]] T&& value() && {
        if (!ok()) {
            throw std::runtime_error("Attempted to access value on error Result: " + status().to_string());
        }
        return std::get<T>(std::move(storage_));
    }

    [[nodiscard]] const Status& status() const noexcept {
        if (ok()) {
            static const Status kOkStatus = Status::ok();
            return kOkStatus;
        }
        return std::get<Status>(storage_);
    }

private:
    std::variant<T, Status> storage_;
};

} // namespace sih26119
