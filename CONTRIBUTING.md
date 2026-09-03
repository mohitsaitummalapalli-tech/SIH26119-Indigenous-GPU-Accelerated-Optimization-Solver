# Contributing Guidelines (SIH26119)

We welcome contributions toward developing this indigenous, high-performance mathematical optimization solver. Please read these guidelines carefully before submitting contributions.

---

## 1. Prime Directive: Zero External Solver Engine Constraint

All contributors must adhere strictly to the fundamental research mandate:

> **DO NOT under any circumstances introduce, link, wrap, embed, or call external optimization solver libraries (such as Gurobi, CPLEX, Xpress, HiGHS, SCIP, CBC, GLPK, OR-Tools solver backends, or NVIDIA cuOpt).**

Any pull request or commit that includes, references, or depends upon third-party solver backends will be rejected immediately.

---

## 2. Coding Standards

- **Language Standard**: Modern ISO **C++20** standard (`-std=c++20`).
- **Code Formatting**: Clean, consistent formatting following standard modern C++ guidelines:
  - 4 spaces per indentation level (no tabs).
  - UpperCamelCase for types and classes (`class SparseMatrix`).
  - snake_case for functions and local variables (`solve_linear_system()`).
  - ALL_CAPS for compile-time constants and macro guards (`MAX_ITERATIONS`).
- **Memory Safety**:
  - Prefer modern RAII mechanisms: `std::unique_ptr`, `std::shared_ptr`, and standard containers (`std::vector`).
  - Avoid raw pointers for ownership semantics; raw pointers are permissible only as non-owning views with documented lifetime invariants.
- **Compiler Warnings**:
  - All code must compile cleanly with zero warnings under `-Wall -Wextra -Wpedantic` (GCC/Clang) and `/W4` (MSVC).

---

## 3. Testing Standards

- Every new module or mathematical routine must be accompanied by comprehensive tests in `tests/`.
- Tests must be registered with **CTest** via `add_test()`.
- Avoid adding third-party testing frameworks unless explicitly agreed upon. Keep tests lightweight, native, and deterministic.
- Tests must verify numerical edge cases: zero division, matrix singularity, empty models, extreme bounds ($-\infty, +\infty$), and degenerate bases.

---

## 4. Git & Commit Guidelines

- **Commit Message Format**: Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
  - `feat: <description>` for new architectural or algorithmic capabilities.
  - `fix: <description>` for bug and numerical fixes.
  - `docs: <description>` for documentation improvements.
  - `chore: <description>` for build, configuration, and repository maintenance.
  - `test: <description>` for adding or updating tests.
  - `refactor: <description>` for code restructuring without changing external contracts.
- **Clean Diff**: Ensure commits do not introduce trailing whitespaces, merge conflicts, or untracked binary/build artifacts.
