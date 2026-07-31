import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/**
 * PromptRevisionDrawer - swipeable panel showing prompt revision history.
 *
 * Shows individual revisions with timestamp, field name, concise diff,
 * parent reference, and a restore action.
 *
 * Properties:
 *   sceneVm: the SceneWorkspaceViewModel
 *   rebuildVm: the RebuildViewModel
 *
 * The drawer is opened from the SceneInspector when the user clicks
 * the history icon next to a prompt field.
 */
Drawer {
    id: drawer
    property var sceneVm: null
    property var rebuildVm: null

    objectName: "promptRevisionDrawer"

    standardOverlay: false
    modal: false
    width: parent.width * 0.7
    implicitHeight: parent.height

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Header
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: headerRow.implicitHeight + 24
            color: "#27272A"
            border.bottom.color: "#3F3F46"
            border.bottom.width: 1

            RowLayout {
                id: headerRow
                anchors.fill: parent
                anchors.margins: 12
                spacing: 12

                Label {
                    text: "Revision History"
                    color: "#F4F4F5"
                    font.pixelSize: 17
                    font.bold: true
                }

                Label {
                    text: sceneVm
                        ? "Scene " + sceneVm.scene_map.selected_scene || "—"
                        : ""
                    color: "#A1A1AA"
                    font.pixelSize: 13
                    Layout.fillWidth: true
                    horizontalAlignment: Text.AlignRight
                }

                Button {
                    objectName: "closeRevisionsButton"
                    text: "✕"
                    palette.button: "#3F3F46"
                    palette.buttonText: "#F4F4F5"
                    Accessible.name: "Close revision history"
                    onClicked: drawer.close()
                }
            }
        }

        // Error banner
        Rectangle {
            objectName: "revisionErrorBanner"
            visible: rebuildVm && !!rebuildVm.error
            color: "#451A03"
            border.color: "#F97316"
            Layout.fillWidth: true
            Layout.preferredHeight: errorLabel.implicitHeight + 16

            Label {
                id: errorLabel
                anchors.centerIn: parent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                text: rebuildVm.error
                color: "#FDBA74"
                wrapMode: Text.Wrap
                font.pixelSize: 13
            }
        }

        // Revision list
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: revisionList
                objectName: "revisionList"
                width: drawer.width - 16
                model: rebuildVm ? rebuildVm.revisions : null
                spacing: 8
                interactive: true
                boundsBehavior: Flickable.StopAtBounds
                clip: true

                delegate: Rectangle {
                    width: revisionList.width
                    height: delegateColumn.implicitHeight + 20
                    color: model.isCurrent ? "#1E1B4B" : "#202024"
                    border.color: model.isCurrent ? "#6366F1" : "#3F3F46"
                    border.width: 1
                    radius: 8

                    ColumnLayout {
                        id: delegateColumn
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 6

                        RowLayout {
                            spacing: 12
                            Layout.fillWidth: true

                            Label {
                                text: model.id ? model.id.substring(0, 8) : ""
                                color: "#818CF8"
                                font.pixelSize: 12
                                font.family: monospace.fontFamily
                            }

                            Label {
                                text: model.timestamp || ""
                                color: "#A1A1AA"
                                font.pixelSize: 12
                            }

                            Label {
                                text: model.restoredFrom ? "↩ restored" : ""
                                color: model.restoredFrom ? "#FBBF24" : "transparent"
                                font.pixelSize: 11
                                Layout.fillWidth: true
                                horizontalAlignment: Text.AlignRight
                            }
                        }

                        Label {
                            text: model.value || "(empty)"
                            color: "#D4D4D8"
                            font.pixelSize: 13
                            wrapMode: Text.Wrap
                            maximumLineCount: 4
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }

                        RowLayout {
                            spacing: 8
                            visible: !(model.isCurrent)

                            Button {
                                objectName: "restoreRevisionButton"
                                text: "Restore"
                                palette.button: "#4F46E5"
                                palette.buttonText: "#FFFFFF"
                                Accessible.name: "Restore revision " + model.id
                                onClicked: {
                                    if (rebuildVm) {
                                        rebuildVm.selected_revision_id = model.id
                                        rebuildVm.restoreSelected()
                                    }
                                }
                            }
                        }

                        // Diff (only show for non-current revisions)
                        TextArea {
                            objectName: "revisionDiff"
                            visible: !!model.diff
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            text: model.diff
                            color: "#A1A1AA"
                            font.pixelSize: 11
                            font.family: monospace.fontFamily
                            Layout.fillWidth: true
                            background: Rectangle {
                                color: "#18181B"
                                radius: 4
                            }
                        }
                    }

                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        height: 1
                        color: "#27272A"
                    }
                }

                Label {
                    anchors.centerIn: parent
                    visible: revisionList.count === 0
                    text: "No revision history available"
                    color: "#71717A"
                    font.pixelSize: 14
                }
            }

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AsNeeded
                contentItem: Rectangle {
                    implicitWidth: 8
                    radius: 4
                    color: "#52525B"
                }
            }
        }
    }
}
