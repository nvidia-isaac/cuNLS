## Contributing to cuNLS

Thank you for your interest in contributing to cuNLS, a CUDA-accelerated library for solving general nonlinear least-squares problems on GPU. This document describes the guidelines for contributing to this project.

### Getting Started

1. Fork the repository and clone your fork.
2. Set up pre-commit hooks (see [Code Style](#code-style) below).
3. Build the library following the instructions in the [README](README.md).
4. Run the test suite to make sure everything passes before making changes.

### Code Style

cuNLS follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Formatting is enforced automatically via `clang-format` (v15) with the configuration in `.clang-format`.

Set up the pre-commit hook after cloning:

```bash
sudo apt install pre-commit
pre-commit install
```

This ensures all committed code is automatically reformatted. To manually reformat files:

```bash
sudo apt install clang-format
find . -iname '*.h' -o -iname '*.cpp' -o -iname '*.cu' | xargs clang-format -i
```

### Pull Request Guidelines

- Keep pull requests focused on a single change or feature.
- Include tests for new functionality or bug fixes.
- Ensure all existing tests pass before submitting.
- Follow the existing code patterns and conventions in the repository.
- Update documentation if your changes affect the public API.

### Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

- Any contribution which contains commits that are not signed-off will not be accepted.

- To sign off on a commit, use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
  Developer Certificate of Origin Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.
  ```

  ```
  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I have
      the right to submit it under the open source license indicated in
      the file; or

  (b) The contribution is based upon previous work that, to the best of
      my knowledge, is covered under an appropriate open source license
      and I have the right under that license to submit that work with
      modifications, whether created in whole or in part by me, under
      the same open source license (unless I am permitted to submit
      under a different license), as indicated in the file; or

  (c) The contribution was provided directly to me by some other person
      who certified (a), (b) or (c) and I have not modified it.

  (d) I understand and agree that this project and the contribution are
      public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

### License

By contributing to cuNLS, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
