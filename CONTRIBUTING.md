# Contributing

Contributions are welcome through focused pull requests.

## Before opening a pull request

1. Create a branch from the current `main` revision.
2. Keep upstream weights, generated `.sabir` images, build output, credentials,
   and board identifiers out of Git.
3. Run:

   ```bash
   python -m pip install -e ".[test]"
   python -m pytest -q
   cmake -S host -B build-host -DCMAKE_BUILD_TYPE=Release
   cmake --build build-host --parallel
   ```

4. For runtime/firmware changes, build with ESP-IDF 5.5.5 for `esp32s3` and
   state whether physical hardware was tested.
5. Update `docs/VALIDATION.md` only for repeatable evidence. Record unsupported
   configurations explicitly; do not generalize a single-board result.

Pull requests should explain the motivation, tests, board and toolchain scope,
format compatibility, flash/RAM impact, and any change to model provenance or
licensing assumptions.

By contributing, you agree that your contribution is licensed under the MIT
license in [LICENSE](LICENSE).
