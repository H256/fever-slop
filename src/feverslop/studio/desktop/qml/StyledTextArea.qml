import QtQuick
import QtQuick.Controls

TextArea {
    id: control

    ContextMenu.menu: null

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.RightButton
        onPressed: contextMenu.popup()
    }

    TextContextMenu {
        id: contextMenu
        textControl: control
    }
}
