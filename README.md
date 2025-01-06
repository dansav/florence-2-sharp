# Florence 2 C# Lab Project

Welcome to the Florence 2 C# Lab Project!

This project explores the requirements and processes involved in performing inference with the Florence 2 models using C#/.NET.

## Getting Started

Follow these instructions to set up your development environment and get started with the project.

All code builds and runs on Windows and macOS.

### Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- **Optional**: To run the Python code
  - [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) Python project manager

## .NET Projects

### Install and Build

1. **Ensure .NET 8.0 SDK is installed**

    Download and install the .NET 8.0 SDK from the [official .NET website](https://dotnet.microsoft.com/download/dotnet/8.0).

2. **Clone the repository:**

    ```sh
    git clone https://github.com/dansav/florence-2-sharp.git
    cd florence-2-sharp
    ```

3. **Build and run the test application**

    ```sh
    dotnet build
    dotnet run --project FlorenceTwoLab.Desktop/FlorenceTwoLab.Desktop.csproj
    ```

## Python Research Project

### Setup

1. Install the [uv](https:?tab=readme-ov-file#installation) tool.

1. Download models and some test data:

    ```sh
    uv run setup.py
    ```

1. Run main test script

    ```sh
    uv run main.py
    ```

## Folder Structure

- [FlorenceTwoLab.Core](FlorenceTwoLab.Core): Core implementation of the Florence 2 model and related utilities
- [FlorenceTwoLab.Core.Tests]: Unit tests for the core functionality
- [FlorenceTwoLab.Desktop]: AvaloniaUI-based desktop application
- [research](research): Research materials and documentation
  - [python_reference_implementation](research/python_reference_implementation): Reference implementation in Python

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the project maintainers.

Happy coding!
