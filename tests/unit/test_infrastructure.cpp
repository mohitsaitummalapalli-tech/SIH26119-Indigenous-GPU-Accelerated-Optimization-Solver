#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <concepts>
#include <limits>
#include <cstdint>

namespace fs = std::filesystem;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "SIH26119 Optimization Solver — Infrastructure Test Suite\n";
    std::cout << "Phase 0: Project Constitution & Repository Foundation\n";
    std::cout << "=========================================================\n";

    int failures = 0;

    // 1. Validate Modern C++20 Feature Support
    std::cout << "[CHECK] Verifying C++20 standard compliance...\n";
    #if __cplusplus < 202002L && (!defined(_MSVC_LANG) || _MSVC_LANG < 202002L)
        std::cerr << "  [FAIL] Compiler is not operating in C++20 mode!\n";
        ++failures;
    #else
        std::cout << "  [PASS] Modern C++20 standard active (__cplusplus: " << __cplusplus << ")\n";
    #endif

    // 2. Validate Standard Numeric Properties
    std::cout << "[CHECK] Verifying numeric properties for scientific computing...\n";
    static_assert(std::numeric_limits<double>::is_iec559, "IEEE 754 floating-point standard is required");
    static_assert(sizeof(double) == 8, "Standard 64-bit IEEE 754 double precision required");
    static_assert(sizeof(int64_t) == 8, "Standard 64-bit integer required for large index spaces");
    std::cout << "  [PASS] IEEE 754 double precision & 64-bit integer sizing verified\n";

    // 3. Deterministic Repository Root Discovery
    std::cout << "[CHECK] Determining repository root deterministically...\n";
    fs::path repo_root;
#ifdef SIH26119_SOURCE_DIR
    repo_root = fs::path(SIH26119_SOURCE_DIR);
#else
    repo_root = fs::current_path();
    if (!fs::exists(repo_root / "src") && fs::exists(repo_root / ".." / "src")) {
        repo_root = repo_root / "..";
    }
    if (!fs::exists(repo_root / "src") && fs::exists(repo_root / ".." / ".." / "src")) {
        repo_root = repo_root / ".." / "..";
    }
#endif

    std::cout << "  [INFO] Repository root resolved at: " << fs::absolute(repo_root).string() << "\n";

    if (!fs::is_directory(repo_root) || !fs::exists(repo_root / "CMakeLists.txt")) {
        std::cerr << "  [FAIL] Invalid repository root: " << repo_root.string() << "\n";
        ++failures;
    } else {
        std::cout << "  [PASS] Valid repository root confirmed (CMakeLists.txt present)\n";
    }

    // 4. Validate Repository Architecture Boundaries (Strict Non-Zero Exit on Missing Directory)
    std::cout << "[CHECK] Verifying architectural source layer boundaries...\n";
    const std::vector<std::string> expected_directories = {
        "src/core",
        "src/model",
        "src/io",
        "src/numerics",
        "src/algorithms",
        "src/backend/cpu",
        "src/backend/cuda",
        "src/verification",
        "src/api",
        "docs/mathematics",
        "docs/algorithms",
        "docs/numerical",
        "docs/gpu",
        "docs/benchmarks",
        "tests/unit",
        "tests/integration",
        "tests/numerical",
        "tests/regression",
        "benchmarks/instances",
        "benchmarks/scripts",
        "benchmarks/results",
        "examples",
        "scripts"
    };

    for (const auto& rel_path : expected_directories) {
        fs::path p = repo_root / rel_path;
        if (fs::is_directory(p)) {
            std::cout << "  [PASS] Found layer: " << rel_path << "\n";
        } else {
            std::cerr << "  [FAIL] Missing required layer: " << rel_path << "\n";
            ++failures;
        }
    }

    std::cout << "=========================================================\n";
    if (failures == 0) {
        std::cout << "INFRASTRUCTURE VERIFICATION PASSED (Phase 0 Foundation OK)\n";
        return 0;
    } else {
        std::cerr << "INFRASTRUCTURE VERIFICATION FAILED (" << failures << " errors)\n";
        return 1;
    }
}
