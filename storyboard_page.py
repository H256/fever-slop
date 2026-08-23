"""CLI wrapper; prefer feverslop.tools.storyboard_page for imports."""

from feverslop.tools.storyboard_page import (
    generate_storyboard_page,
    main,
    parse_scene_list,
)

__all__ = ["generate_storyboard_page", "main", "parse_scene_list"]

if __name__ == "__main__":
    main()
