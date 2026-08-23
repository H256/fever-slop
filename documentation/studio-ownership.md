# Studio Package Retirement

The deprecated `feverslop.studio` Python package was removed under umbrella
issue #600. Headless services now live in `application/`, `composition/`, or
`adapters/` according to their responsibility.

The `.studio/` directory inside a project is unrelated to the retired Python
package. Its metadata and pipeline-state files remain supported on-disk
formats.
