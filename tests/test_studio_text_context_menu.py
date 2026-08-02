from __future__ import annotations

import re
import unittest
from pathlib import Path


QML_ROOT = Path(__file__).parents[1] / "src" / "feverslop" / "studio" / "desktop" / "qml"


class StudioTextContextMenuTests(unittest.TestCase):
    def test_text_controls_use_the_custom_qml_context_menu(self) -> None:
        menu = QML_ROOT / "TextContextMenu.qml"
        text_area = QML_ROOT / "StyledTextArea.qml"
        text_field = QML_ROOT / "StyledTextField.qml"

        self.assertTrue(menu.is_file())
        self.assertTrue(text_area.is_file())
        self.assertTrue(text_field.is_file())

        menu_source = menu.read_text(encoding="utf-8")
        self.assertIn("popupType: Popup.Item", menu_source)
        self.assertIn("#F4F4F5", menu_source)
        self.assertIn("#71717A", menu_source)
        self.assertIn("#3F3F46", menu_source)

        for control in (text_area, text_field):
            source = control.read_text(encoding="utf-8")
            self.assertIn("MouseArea", source)
            self.assertIn("acceptedButtons: Qt.RightButton", source)
            self.assertIn("contextMenu.popup()", source)

        for path in QML_ROOT.rglob("*.qml"):
            if path.name in {"StyledTextArea.qml", "StyledTextField.qml"}:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertIsNone(
                re.search(r"(?<!Styled)Text(?:Area|Field)\\s*\\{", source),
                f"{path.relative_to(QML_ROOT)} bypasses the styled text controls",
            )


if __name__ == "__main__":
    unittest.main()
