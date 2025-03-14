# AutoAgent Development Guide

## Build & Test Commands
- **Install**: `pip install -e .` (development mode)
- **Run CLI**: `auto` (main entry point)
- **Run Tests**: `pytest` (all tests)
- **Run Single Test**: `pytest tests/path_to_test.py::test_function_name -v`
- **Documentation**: 
  - Build: `cd docs && npm run build`
  - Serve: `cd docs && npm run serve`

## Code Style Guidelines
- **Formatting**: Follows PEP 8 with 120 character line limit (autopep8 aggressive level 3)
- **Imports**: Group standard library, third-party, and project imports separately
- **Types**: Use type hints for function parameters and return values
- **Naming**:
  - Classes: `CamelCase`
  - Functions/Methods: `snake_case`
  - Constants: `UPPER_CASE`
- **Error Handling**: Use specific exceptions with descriptive messages
- **Documentation**: Follow guidelines in `docs/DOC_STYLE_GUIDE.md` for clarity and conciseness
- **Comments**: Explain "why" not "what" in comments

## Project Structure
- Agent implementations in `autoagent/agents/`
- Tools in `autoagent/tools/`
- Core workflow and execution in `autoagent/flow/`