import QtQuick
import QtQuick.Controls

Menu {
    id: menu

    required property var textControl

    popupType: Popup.Window
    implicitWidth: 216
    padding: 4

    background: Rectangle {
        color: "#27272A"
        border.color: "#52525B"
        border.width: 1
        radius: 6
    }

    component TextAction: MenuItem {
        id: item

        implicitWidth: 208
        implicitHeight: 34
        leftPadding: 12
        rightPadding: 12

        contentItem: Text {
            text: item.text
            color: item.enabled ? "#F4F4F5" : "#71717A"
            font: item.font
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            color: item.highlighted ? "#3F3F46" : "transparent"
            radius: 4
        }
    }

    TextAction {
        text: qsTr("Undo")
        enabled: menu.textControl.canUndo
        onTriggered: menu.textControl.undo()
    }
    TextAction {
        text: qsTr("Redo")
        enabled: menu.textControl.canRedo
        onTriggered: menu.textControl.redo()
    }
    MenuSeparator {
        contentItem: Rectangle {
            implicitHeight: 1
            color: "#52525B"
        }
    }
    TextAction {
        text: qsTr("Cut")
        enabled: !menu.textControl.readOnly && menu.textControl.selectedText.length > 0
        onTriggered: menu.textControl.cut()
    }
    TextAction {
        text: qsTr("Copy")
        enabled: menu.textControl.selectedText.length > 0
        onTriggered: menu.textControl.copy()
    }
    TextAction {
        text: qsTr("Paste")
        enabled: !menu.textControl.readOnly && menu.textControl.canPaste
        onTriggered: menu.textControl.paste()
    }
    TextAction {
        text: qsTr("Delete")
        enabled: !menu.textControl.readOnly && menu.textControl.selectedText.length > 0
        onTriggered: menu.textControl.remove(menu.textControl.selectionStart, menu.textControl.selectionEnd)
    }
    MenuSeparator {
        contentItem: Rectangle {
            implicitHeight: 1
            color: "#52525B"
        }
    }
    TextAction {
        text: qsTr("Select All")
        enabled: menu.textControl.length > 0
        onTriggered: menu.textControl.selectAll()
    }
}
