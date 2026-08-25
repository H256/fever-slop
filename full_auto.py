from feverslop.cli.full_auto import (
    build_arg_parser,
    build_full_auto_use_case,  # noqa: F401
    console,
    coerce_local_path,  # noqa: F401
    FullAutoRequest,  # noqa: F401
    parse_optional_bool,  # noqa: F401
    request_from_args,  # noqa: F401
    run_full_auto_command,
)  # noqa: F401


def main() -> None:
    run_full_auto_command(build_arg_parser().parse_args(), output=console)


if __name__ == "__main__":
    main()
