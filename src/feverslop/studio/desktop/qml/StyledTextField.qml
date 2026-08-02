import QtQuick
import QtQuick.Controls

TextField {
    id: control

    ContextMenu.menu: contextMenu

    TextContextMenu {
        id: contextMenu
        textControl: control
    }
}
