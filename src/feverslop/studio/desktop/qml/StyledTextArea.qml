import QtQuick
import QtQuick.Controls

TextArea {
    id: control

    ContextMenu.menu: null

    TapHandler {
        acceptedButtons: Qt.RightButton
        onPressedChanged: {
            if (pressed === (Application.styleHints.contextMenuTrigger === Qt.ContextMenuTrigger.Press)) {
                contextMenu.popup()
            }
        }
    }

    TextContextMenu {
        id: contextMenu
        textControl: control
    }
}
