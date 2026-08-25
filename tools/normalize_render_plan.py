"""Legacy facade for the packaged render-plan normalizer CLI."""

from feverslop.tools import normalize_render_plan as _cli
from feverslop.tools.normalize_render_plan import normalize_render_plan_file
from rich.console import Console

console = Console()
__all__ = ["console", "main", "normalize_render_plan_file"]


def main():
    _cli.console = console
    _cli.normalize_render_plan_file = normalize_render_plan_file
    return _cli.main()

if __name__ == "__main__":
    main()
