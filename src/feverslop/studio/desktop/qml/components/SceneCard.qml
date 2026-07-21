import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Control {
    id: card
    required property int sceneNumber
    required property real startSeconds
    required property real endSeconds
    required property string performanceState
    required property string status
    required property url thumbnailUrl
    required property bool selected
    signal activated(int sceneNumber)

    objectName: "sceneCard_" + sceneNumber
    focusPolicy: Qt.StrongFocus
    Accessible.role: Accessible.ListItem
    Accessible.name: "Scene " + sceneNumber + ", " + status
    Accessible.description: (endSeconds - startSeconds).toFixed(1) + " seconds, " + performanceState
    Keys.onSpacePressed: activated(sceneNumber)
    Keys.onReturnPressed: activated(sceneNumber)
    Keys.onEnterPressed: activated(sceneNumber)
    padding: 8

    background: Rectangle {
        color: card.selected ? "#ECECFF" : card.hovered ? "#F2F2F7" : "#FFFFFF"
        border.color: card.activeFocus ? "#5B5FC7" : card.selected ? "#777BE0" : "#D8D8DC"
        border.width: card.activeFocus || card.selected ? 2 : 1
        radius: 7
    }

    contentItem: RowLayout {
        spacing: 10

        Rectangle {
            Layout.preferredWidth: 96
            Layout.preferredHeight: 54
            color: "#E5E5EA"
            radius: 4
            clip: true

            Image {
                id: previewImage
                anchors.fill: parent
                source: card.thumbnailUrl
                fillMode: Image.PreserveAspectCrop
                asynchronous: true
                visible: status === Image.Ready
            }
            Label {
                anchors.centerIn: parent
                text: "No preview"
                visible: previewImage.status !== Image.Ready
                color: "#6E6E73"
                font.pixelSize: 11
            }
        }

        ColumnLayout {
            spacing: 3
            Layout.fillWidth: true
            Label { text: "Scene " + card.sceneNumber; color: "#1C1C1E"; font.bold: true }
            Label {
                text: card.startSeconds.toFixed(1) + " - " + card.endSeconds.toFixed(1) + " s"
                color: "#6E6E73"
                font.pixelSize: 12
            }
            Label {
                text: card.performanceState || "No performance state"
                color: "#48484A"
                font.pixelSize: 12
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
        }

        Label {
            text: card.status || "unknown"
            color: card.status === "failed" ? "#C62828" : "#48484A"
            font.pixelSize: 11
            font.bold: true
        }
    }

    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        onClicked: {
            card.forceActiveFocus()
            card.activated(card.sceneNumber)
        }
    }
}
