import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: page
    clip: true
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null

    ColumnLayout {
        width: page.availableWidth
        spacing: 18
        anchors.margins: 28

        RowLayout {
            Layout.fillWidth: true
            Label { text: "Local project folders"; color: "#6E6E73"; font.pixelSize: 15; Layout.fillWidth: true }
            Button {
                text: "Create Project"
                icon.name: "list-add"
                implicitHeight: 44
                palette.buttonText: "#FFFFFF"
                background: Rectangle { color: parent.hovered ? "#666AD1" : "#5B5FC7"; radius: 6 }
                onClicked: createDialog.open()
            }
        }

        GridView {
            id: grid
            Layout.fillWidth: true
            Layout.preferredHeight: Math.max(280, Math.ceil(count / Math.max(1, Math.floor(width / 300))) * 156)
            cellWidth: Math.max(280, width / Math.max(1, Math.floor(width / 300)))
            cellHeight: 156
            model: vm ? vm.projects : []
            delegate: Item {
                required property var modelData
                width: grid.cellWidth
                height: grid.cellHeight
                Rectangle {
                    anchors.fill: parent
                    anchors.margins: 6
                    color: mouse.containsMouse ? "#F5F5F7" : "#FFFFFF"
                    border.color: "#D8D8DC"
                    radius: 6
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 18
                        Label { text: modelData.name; color: "#1C1C1E"; font.bold: true; font.pixelSize: 17; elide: Text.ElideRight; Layout.fillWidth: true }
                        Label { text: modelData.id; color: "#6E6E73"; font.pixelSize: 13; elide: Text.ElideMiddle; Layout.fillWidth: true }
                        Item { Layout.fillHeight: true }
                        Label { text: (modelData.project_type || "standard_music_video").replaceAll("_", " "); color: "#5B5FC7"; font.pixelSize: 12 }
                    }
                    MouseArea {
                        id: mouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            if (vm) vm.select_project(modelData.id)
                            root.currentPage = 1
                            root.pageTitle = "Dashboard"
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: createDialog
        modal: true
        anchors.centerIn: Overlay.overlay
        width: 520
        title: "Create Project"
        standardButtons: Dialog.Cancel
        ColumnLayout {
            width: parent.width
            spacing: 12
            Label { text: "Project creation is available after the project workflow is loaded."; wrapMode: Text.WordWrap; Layout.fillWidth: true }
        }
    }
}
