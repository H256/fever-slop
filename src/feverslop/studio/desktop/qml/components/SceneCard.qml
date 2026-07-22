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
    property bool keyboardCurrent: false
    signal activated(int sceneNumber)

    objectName: "sceneCard_" + sceneNumber
    Accessible.role: Accessible.ListItem
    Accessible.name: "Scene " + sceneNumber + ", " + status
    Accessible.description: (endSeconds - startSeconds).toFixed(1) + " seconds, " + performanceState
    Accessible.checkable: true
    Accessible.checked: selected
    padding: 12

    background: Rectangle {
        color: card.selected ? "#312E59" : card.hovered ? "#303036" : "#27272A"
        border.color: card.activeFocus || card.keyboardCurrent ? "#A5B4FC" : card.selected ? "#818CF8" : "#3F3F46"
        border.width: card.activeFocus || card.keyboardCurrent || card.selected ? 2 : 1
        radius: 10
    }

    contentItem: RowLayout {
        spacing: 14

        Rectangle {
            Layout.preferredWidth: 144
            Layout.preferredHeight: 81
            color: "#18181B"
            border.color: "#3F3F46"
            radius: 7
            clip: true

            Image {
                id: previewImage
                objectName: "scenePreviewImage_" + card.sceneNumber
                anchors.fill: parent
                source: card.thumbnailUrl
                fillMode: Image.PreserveAspectCrop
                asynchronous: true
                visible: status === Image.Ready
            }
            Label {
                anchors.centerIn: parent
                text: card.thumbnailUrl ? "Loading preview..." : "No preview"
                visible: previewImage.status !== Image.Ready
                color: "#A1A1AA"
                font.pixelSize: 12
            }
        }

        ColumnLayout {
            spacing: 6
            Layout.fillWidth: true
            Label { text: "Scene " + card.sceneNumber; color: "#F4F4F5"; font.pixelSize: 15; font.bold: true }
            Label {
                text: card.startSeconds.toFixed(1) + " - " + card.endSeconds.toFixed(1) + " s"
                color: "#A1A1AA"
                font.pixelSize: 13
            }
            Label {
                text: card.performanceState || "No performance state"
                color: "#D4D4D8"
                font.pixelSize: 13
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
        }

        Label {
            text: card.status || "unknown"
            color: card.status === "failed" ? "#F87171" : "#D4D4D8"
            font.pixelSize: 12
            font.bold: true
        }
    }

    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        onClicked: card.activated(card.sceneNumber)
    }
}
