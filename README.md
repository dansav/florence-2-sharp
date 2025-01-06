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

1. Install the [uv](https:?tab=readme-ov-file#installation//github.1. Download models and some test data:

    ```sh
    uv run setup.py
    ```

1. Run main test script

    ```sh
    uv run main.py
    ```
com/astral-sh/uv?tab=readme-ov-file#installation)FlorenceTwoLab.Coreand some test data:

    ```sh
    uv run setup.py
    ```

1. Run FlorenceTwoLab.Core.Testssh
    uv run main.py
    ```

## Folder Structure

- [FlorenceTwoLFlorenceTwoLab.Desktope): Contains the core project source code.
- [FlorenceTwoLab.Core.Tests](FlorenceTwoLab.Core.Tests): Contains researchject.
- [FlorenceTwoLab.Desktop](FlorenceTwoLab.  - [python_reference_implementation](research/python_reference_implementation) The reference implementation using Python.
Desktop): Contains the desktop application source code built with AvaloniaUI and CommunityToolkit MVVM.
- [research](research): Contains stubs and hints on data structures.
  - [python_referen all image-related tasks.
- **AvaloniaUI** and **CommunityToolkit MVVM** framework for building the user in6erface.

## License

This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/6) file for details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the project maintainers.

Happy coding!
