import QtQuick
import QtQuick.Controls

TextArea {
    id: control

    ContextMenu.menu: contextMenu

    TextContextMenu {
        id: contextMenu
        textControl: control
    }
}
