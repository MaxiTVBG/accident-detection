# Contributing to Accident Detection & AI Analysis

First off, thank you for considering contributing to this project! Any contribution, whether it's a bug report, a new feature, or a documentation improvement, is greatly appreciated.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue and provide the following information:

-   A clear and descriptive title.
-   A detailed description of the problem, including steps to reproduce it.
-   Any relevant error messages or logs.
-   Your operating system and Python version.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please create an issue to discuss it. This allows us to coordinate efforts and ensure the proposed change aligns with the project's goals.

### Pull Requests

We welcome pull requests! If you'd like to contribute code, please follow these steps:

1.  **Fork the repository** and create a new branch from `main`.
2.  **Set up your development environment:**
    ```bash
    # Install main dependencies
    pip install -r requirements.txt
    # Install development/testing tools
    pip install black flake8 pytest mypy
    ```
3.  **Make your changes.** Ensure your code adheres to the project's style.
4.  **Format and lint your code** before committing:
    ```bash
    # Auto-format your code
    black .
    # Check for linting issues
    flake8 .
    ```
5.  **Add or update tests** for your changes. We use `pytest` for testing.
    ```bash
    # Run the test suite
    pytest
    ```
6.  **Ensure all tests pass.**
7.  **Create a pull request** with a clear description of your changes.

## Code of Conduct

This project and everyone participating in it is governed by a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. (Note: A `CODE_OF_CONDUCT.md` file would need to be created separately).

Thank you for your contribution!
