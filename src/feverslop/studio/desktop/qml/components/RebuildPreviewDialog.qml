import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/**
 * RebuildPreviewDialog - modal dialog showing rebuild impact before confirmation.
 */
Dialog {
    id: dialog
    property var studioVm: null
    property var sceneVm: null
    readonly property bool conflictingJob: {
        if (!studioVm || !studioVm.active_job)
            return false
        const status = studioVm.active_job.status || ""
        return status === "active" || status === "queued" || status === "running"
    }
    readonly property bool acceptDisabled: conflictingJob

    objectName: "rebuildPreviewDialog"
    modal: true
    dim: true
    closePolicy: Popup.CloseOnEscape

    title: "Rebuild Preview"

    width: Math.min(parent.width * 0.8, 700)
        + ((reuseSection.height > 0 || rebuildSection.height > 0
            || invalidateSection.height > 0 || unknownSection.height > 0) ? 0 : 200)

    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        Label {
            objectName: "rebuildDescription"
            text: "The following artifacts will be affected by the rebuild:"
            color: "#D4D4D8"
            font.pixelSize: 14
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        // Reusable artifacts - green
        Rectangle {
            id: reuseSection
            color: "#052E16"
            border.color: "#22C55E"
            radius: 8
            Layout.fillWidth: true
            visible: reuseSection.children.length > 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    spacing: 8
                    Layout.fillWidth: true

                    Label {
                        text: "\u2713"
                        color: "#22C55E"
                        font.pixelSize: 16
                        font.bold: true
                    }

                    Label {
                        text: "Reuse"
                        color: "#F4F4F5"
                        font.pixelSize: 14
                        font.bold: true
                    }

                    Label { Layout.fillWidth: true }
                }

                Item {
                    id: reuseHolder
                    Layout.fillWidth: true
                    property var kinds: []
                    property var scenes: []
                }
            }
        }

        // Rebuild artifacts - blue
        Rectangle {
            id: rebuildSection
            color: "#1E3A5F"
            border.color: "#60A5FA"
            radius: 8
            Layout.fillWidth: true
            visible: rebuildHolder.kinds.length > 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    spacing: 8
                    Layout.fillWidth: true

                    Label {
                        text: "⟳"
                        color: "#60A5FA"
                        font.pixelSize: 16
                        font.bold: true
                    }

                    Label {
                        text: "Rebuild"
                        color: "#F4F4F5"
                        font.pixelSize: 14
                        font.bold: true
                    }

                    Label { Layout.fillWidth: true }
                }

                Item {
                    id: rebuildHolder
                    Layout.fillWidth: true
                    property var kinds: []
                    property var scenes: []
                }
            }
        }

        // Invalidated artifacts - orange
        Rectangle {
            id: invalidateSection
            color: "#451A03"
            border.color: "#FB923C"
            radius: 8
            Layout.fillWidth: true
            visible: invalidateHolder.kinds.length > 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    spacing: 8
                    Layout.fillWidth: true

                    Label {
                        text: "⚠"
                        color: "#FB923C"
                        font.pixelSize: 16
                        font.bold: true
                    }

                    Label {
                        text: "Invalidate"
                        color: "#F4F4F5"
                        font.pixelSize: 14
                        font.bold: true
                    }

                    Label { Layout.fillWidth: true }
                }

                Item {
                    id: invalidateHolder
                    Layout.fillWidth: true
                    property var kinds: []
                    property var scenes: []
                }
            }
        }

        // Unknown legacy artifacts - gray
        Rectangle {
            id: unknownSection
            color: "#1E293B"
            border.color: "#94A3B8"
            radius: 8
            Layout.fillWidth: true
            visible: unknownHolder.kinds.length > 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    spacing: 8
                    Layout.fillWidth: true

                    Label {
                        text: "?"
                        color: "#94A3B8"
                        font.pixelSize: 16
                        font.bold: true
                    }

                    Label {
                        text: "Unknown Legacy"
                        color: "#F4F4F5"
                        font.pixelSize: 14
                        font.bold: true
                    }

                    Label { Layout.fillWidth: true }

                    CheckBox {
                        visible: true
                        text: "I understand these artifacts lack provenance"
                        checked: false
                        contentItem: Text {
                            text: parent.text
                            color: "#94A3B8"
                        }
                    }
                }
            }
        }

        // Summary
        Rectangle {
            objectName: "rebuildSummary"
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 8
            Layout.fillWidth: true
            Layout.preferredHeight: summaryColumn.implicitHeight + 20

            ColumnLayout {
                id: summaryColumn
                anchors.fill: parent
                anchors.margins: 14
                spacing: 6

                Label {
                    text: "Total: "
                    + (reuseHolder.kinds.length
                       + rebuildHolder.kinds.length
                       + invalidateHolder.kinds.length
                       + unknownHolder.kinds.length)
                    + " artifact types affected"
                }
            }
        }

        // Warning for conflicting job
        Rectangle {
            visible: dialog.conflictingJob
            color: "#451A03"
            border.color: "#F97316"
            radius: 6
            Layout.fillWidth: true
            Layout.preferredHeight: conflictLabel.implicitHeight + 16

            Label {
                id: conflictLabel
                anchors.centerIn: parent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                text: "⚠ A job is currently running. Rebuild preview may be stale."
                color: "#FDBA74"
                wrapMode: Text.Wrap
            }
        }
    }

    standardButtons: Dialog.Cancel | Dialog.Apply

    onApplied: {
        if (!dialog.acceptDisabled) {
            dialog.rebuildAccepted()
            dialog.close()
        }
    }

    Button {
        visible: dialog.acceptDisabled
        enabled: false
        text: "Run"
        palette.button: "#3F3F46"
        palette.buttonText: "#71717A"
    }

    signal rebuildAccepted()

    function _setSections(reuse, rebuild, invalidate, unknown) {
        reuseHolder.kinds = reuse
        rebuildHolder.kinds = rebuild
        invalidateHolder.kinds = invalidate
        unknownHolder.kinds = unknown
    }
}
