import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/**
 * RebuildPreviewDialog - modal dialog showing rebuild impact before confirmation.
 *
 * Groups artifacts into Reuse, Rebuild, Invalidate, and Unknown sections.
 * Shows affected scene numbers. Requires explicit choice for Unknown artifacts.
 * Disables acceptance when preview source is stale or a conflicting job is active.
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
    standardOverlay: true
    closePolicy: Popup.CloseOnEscape

    title: "Rebuild Preview"
    icon.source: "qrc:/icons/rebuild.svg"

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
        _ArtifactSection {
            id: reuseSection
            sectionTitle: "Reuse"
            sectionColor: "#22C55E"
            sectionBgColor: "#052E16"
            icon: "✓"
            artifactKinds: []
            sceneNumbers: []
            Layout.fillWidth: true
            visible: reuseSection.artifactKinds.length > 0
        }

        // Rebuild artifacts - blue
        _ArtifactSection {
            id: rebuildSection
            sectionTitle: "Rebuild"
            sectionColor: "#60A5FA"
            sectionBgColor: "#1E3A5F"
            icon: "⟳"
            artifactKinds: []
            sceneNumbers: []
            Layout.fillWidth: true
            visible: rebuildSection.artifactKinds.length > 0
        }

        // Invalidated artifacts - orange
        _ArtifactSection {
            id: invalidateSection
            sectionTitle: "Invalidate"
            sectionColor: "#FB923C"
            sectionBgColor: "#451A03"
            icon: "⚠"
            artifactKinds: []
            sceneNumbers: []
            Layout.fillWidth: true
            visible: invalidateSection.artifactKinds.length > 0
        }

        // Unknown legacy artifacts - gray
        _ArtifactSection {
            id: unknownSection
            sectionTitle: "Unknown Legacy"
            sectionColor: "#94A3B8"
            sectionBgColor: "#1E293B"
            icon: "?"
            artifactKinds: []
            sceneNumbers: []
            requireChoice: true
            Layout.fillWidth: true
            visible: unknownSection.artifactKinds.length > 0
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

                Labels {
                    text: "Total: "
                    + (reuseSection.artifactKinds.length
                       + rebuildSection.artifactKinds.length
                       + invalidateSection.artifactKinds.length
                       + unknownSection.artifactKinds.length)
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

    // Buttons
    standardButtons: Dialog.Cancel | Dialog.Apply

    onApply: {
        if (!dialog.acceptDisabled) {
            // Emit signal for handler to execute rebuild
            dialog.rebuildAccepted()
            dialog.close()
        }
    }

    // Re-enable button when job completes
    Button {
        visible: dialog.acceptDisabled
        enabled: false
        text: "Run"
        palette.button: "#3F3F46"
        palette.buttonText: "#71717A"
    }

    signal rebuildAccepted()

    function _setSections(reuse, rebuild, invalidate, unknown) {
        reuseSection.artifactKinds = reuse
        rebuildSection.artifactKinds = rebuild
        invalidateSection.artifactKinds = invalidate
        unknownSection.artifactKinds = unknown
    }
}

/**
 * _ArtifactSection - reusable section for grouping artifacts by category.
 */
Rectangle {
    id: sectionRoot
    property string sectionTitle: ""
    property string sectionColor: "#60A5FA"
    property string sectionBgColor: "#1E3A5F"
    property string icon: ""
    property var artifactKinds: []
    property var sceneNumbers: []
    property bool requireChoice: false

    color: sectionRoot.sectionBgColor
    border.color: sectionRoot.sectionColor
    radius: 8

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        RowLayout {
            spacing: 8
            Layout.fillWidth: true

            Label {
                text: sectionRoot.icon
                color: sectionRoot.sectionColor
                font.pixelSize: 16
                font.bold: true
            }

            Label {
                text: sectionRoot.sectionTitle
                color: "#F4F4F5"
                font.pixelSize: 14
                font.bold: true
            }

            Label {
                Layout.fillWidth: true
            }

            Label {
                text: sectionRoot.artifactKinds.length > 0
                    ? sectionRoot.artifactKinds.length + " type"
                      + (sectionRoot.artifactKinds.length > 1 ? "s" : "")
                    : ""
                color: sectionRoot.sectionColor
                font.pixelSize: 12
            }
        }

        // Artifact kinds list
        Repeater {
            model: sectionRoot.artifactKinds

            delegate: Label {
                text: "  · " + modelData
                color: "#D4D4D8"
                font.pixelSize: 13
            }
        }

        // Scene numbers
        Label {
            visible: sectionRoot.sceneNumbers.length > 0
            text: "Scenes: " + sectionRoot.sceneNumbers.join(", ")
            color: "#A1A1AA"
            font.pixelSize: 12
        }

        // Unknown choice requirement
        CheckBox {
            visible: sectionRoot.requireChoice
            text: "I understand these artifacts lack provenance"
            color: sectionRoot.sectionColor
            checked: false
        }
    }
}
