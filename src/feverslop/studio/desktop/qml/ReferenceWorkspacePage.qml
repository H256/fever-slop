import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

Rectangle {
    id: page
    color: "#18181B"
    objectName: "referenceWorkspacePage"

    property var refVm: typeof referenceWorkspaceViewModel !== "undefined" ? referenceWorkspaceViewModel : null
    property var studioVm: typeof studioViewModel !== "undefined" ? studioViewModel : null

    readonly property bool hasLibrary: refVm && refVm.library_model && refVm.has_projects
    readonly property bool hasSelection: refVm && refVm.selected_asset !== ""

    Component.onCompleted: {
        if (refVm && studioVm && studioVm.current_project_id) {
            refVm.set_project(studioVm.current_project_id)
        }
    }

    Connections {
        target: page.studioVm
        function onCurrentProjectChanged() {
            if (page.refVm && page.studioVm && page.studioVm.current_project_id) {
                page.refVm.set_project(page.studioVm.current_project_id)
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 16

        // Error banner
        Label {
            visible: page.refVm && !!page.refVm.error_message
            text: page.refVm ? page.refVm.error_message : ""
            color: "#8B1A1A"
            font.bold: true
            wrapMode: Text.Wrap
            padding: 10
            Layout.fillWidth: true
            background: Rectangle {
                color: "#FDECEC"
                border.color: "#C62828"
                radius: 6
            }
        }

        // Status message
        Label {
            visible: page.refVm && !!page.refVm.status_message && !page.refVm.error_message
            text: page.refVm ? page.refVm.status_message : ""
            color: "#0F5132"
            wrapMode: Text.Wrap
            padding: 10
            Layout.fillWidth: true
            background: Rectangle {
                color: "#D1E7DD"
                border.color: "#0F5132"
                radius: 6
            }
        }

        // Toolbar
        Rectangle {
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.preferredHeight: toolbarRow.implicitHeight + 20

            RowLayout {
                id: toolbarRow
                anchors.fill: parent
                anchors.margins: 8
                spacing: 10

                ComboBox {
                    id: kindFilter
                    objectName: "kindFilterCombo"
                    model: ["All kinds", "actor", "location", "background", "style"]
                    currentIndex: 0
                    palette.windowText: "#F4F4F5"
                    palette.button: "#353539"
                    palette.window: "#27272A"
                    palette.placeholderText: "#8E8E93"
                    onActivated: page.refVm.set_filter_kind(currentText === "All kinds" ? "" : currentText)
                }

                Switch {
                    id: staleSwitch
                    objectName: "staleOnlySwitch"
                    text: "Stale only"
                    checked: false
                    palette.text: "#C7C7CC"
                    onToggled: page.refVm.set_stale_only(checked)
                }

                Switch {
                    id: missingSwitch
                    objectName: "missingOnlySwitch"
                    text: "Missing only"
                    checked: false
                    palette.text: "#C7C7CC"
                    onToggled: page.refVm.set_missing_only(checked)
                }

                StyledTextField {
                    id: searchField
                    objectName: "refSearchField"
                    placeholderText: "Search assets..."
                    color: "#F4F4F5"
                    placeholderTextColor: "#A1A1AA"
                    palette.button: "#353539"
                    Layout.fillWidth: true
                }

                Button {
                    id: refreshBtn
                    objectName: "refreshRefsBtn"
                    text: "Refresh"
                    display: AbstractButton.TextOnly
                    palette.buttonText: "#A6A6AB"
                    Accessible.name: "Refresh reference library"
                    onClicked: {
                        if (page.refVm) page.refVm.set_project(page.studioVm.current_project_id)
                    }
                }

                Button {
                    id: importBtn
                    objectName: "importAssetBtn"
                    text: "Import"
                    palette.buttonText: "#F4F4F5"
                    background: Rectangle {
                        color: "#5B5FC7"
                        radius: 6
                    }
                    Accessible.name: "Import reference asset"
                    onClicked: {
                        // Placeholder - file dialog integration
                        if (page.refVm) page.refVm.status_message = "Import dialog placeholder"
                    }
                }
            }
        }

        // Main content
        RowLayout {
            spacing: 16
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Asset list
            Rectangle {
                color: "#27272A"
                border.color: "#3F3F46"
                radius: 10
                Layout.preferredWidth: 360
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 3
                    spacing: 0

                    Label {
                        text: "Reference Library"
                        color: "#F4F4F5"
                        font.bold: true
                        font.pixelSize: 14
                        Layout.fillWidth: true
                        Layout.topMargin: 8
                        Layout.leftMargin: 12
                    }

                    ListView {
                        id: refList
                        objectName: "referenceListView"
                        model: page.refVm ? page.refVm.library_model : null
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        ScrollBar.vertical: ScrollBar {}

                        delegate: Rectangle {
                            width: refList.width - 6
                            height: contentRow.implicitHeight + 16
                            color: (mouseArea.containsMouse ? "#353539" :
                                    (refVm.selected_asset === model.id ? "#3A429B" : "transparent"))
                            radius: 6

                            RowLayout {
                                id: contentRow
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 8

                                Image {
                                    visible: model.thumbnailUrl !== ""
                                    source: model.thumbnailUrl
                                    width: 48
                                    height: 48
                                    fillMode: Image.PreserveAspectCrop
                                    layer.enabled: true
                                    layer.smooth: true
                                    cache: true
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 2

                                    Label {
                                        text: model.label
                                        color: "#F4F4F5"
                                        font.bold: true
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }

                                        Label {
                                            text: model.kind + (model.source ? " - " + model.source : "")
                                            color: "#8E8E93"
                                            font.pixelSize: 11
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                        }

                                        ArtifactFreshnessBadge {
                                            status: model.exists ? (model.stale ? "stale" : "current") : "unknown"
                                        }
                                }
                            }

                            MouseArea {
                                id: mouseArea
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: {
                                    page.refVm.select_asset(model.id)
                                }
                            }
                        }
                    }

                    Label {
                        visible: !(refVm && refVm.library_model) || refVm.library_model.rowCount < 1
                        text: "No assets found"
                        color: "#666668"
                        font.italic: true
                        Layout.alignment: Qt.AlignHCenter
                    }
                }
            }

            // Right panel
            ColumnLayout {
                spacing: 16
                Layout.fillWidth: true
                Layout.fillHeight: true

                // Generation panel
                Rectangle {
                    color: "#27272A"
                    border.color: "#3F3F46"
                    radius: 10
                    Layout.fillWidth: true
                    Layout.preferredHeight: generationColumn.implicitHeight + 24

                    ColumnLayout {
                        id: generationColumn
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 10

                        Label {
                            text: "Generation"
                            color: "#F4F4F5"
                            font.bold: true
                            font.pixelSize: 15
                            Layout.fillWidth: true
                        }

                        RowLayout {
                            spacing: 8
                            Layout.fillWidth: true

                            Button {
                                text: "Ingredients"
                                display: AbstractButton.TextOnly
                                palette.buttonText: "#A6A6AB"
                                onClicked: {
                                    page.refVm.queue_generation("ingredients", {})
                                }
                            }

                            Button {
                                text: "Storyboard"
                                display: AbstractButton.TextOnly
                                palette.buttonText: "#A6A6AB"
                                onClicked: {
                                    page.refVm.queue_generation("storyboard", {})
                                }
                            }
                        }
                    }
                }

                // Asset details (when selected)
                Rectangle {
                    color: "#27272A"
                    border.color: "#3F3F46"
                    radius: 10
                    Layout.fillWidth: true
                    Layout.preferredHeight: detailsColumn.implicitHeight + 24
                    visible: page.hasSelection

                    ColumnLayout {
                        id: detailsColumn
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 10

                        Label {
                            text: "Asset Details"
                            color: "#F4F4F5"
                            font.bold: true
                            font.pixelSize: 15
                            Layout.fillWidth: true
                        }

                        Label {
                            text: "ID: " + (refVm.selected_asset_info.id || "")
                            color: "#C7C7CC"
                            font.pixelSize: 12
                            Layout.fillWidth: true
                        }

                        Label {
                            text: "Kind: " + (refVm.selected_asset_info.kind || "")
                            color: "#C7C7CC"
                            font.pixelSize: 12
                            Layout.fillWidth: true
                        }

                        Label {
                            text: "Label: " + (refVm.selected_asset_info.label || "")
                            color: "#C7C7CC"
                            font.pixelSize: 12
                            Layout.fillWidth: true
                        }

                        Label {
                            text: "Dimensions: " + (refVm.selected_asset_info.width || 0) + "x" + (refVm.selected_asset_info.height || 0)
                            color: "#C7C7CC"
                            font.pixelSize: 12
                            Layout.fillWidth: true
                        }

                        Label {
                            text: "Status: " +
                                (refVm.selected_asset_info.exists ? "exists" : "missing") +
                                (refVm.selected_asset_info.stale ? " - stale" : " - fresh")
                            color: refVm.selected_asset_info.exists ? "#C7C7CC" : "#C62828"
                            font.pixelSize: 12
                            Layout.fillWidth: true
                        }

                        ArtifactFreshnessBadge {
                            status: refVm.selected_asset_info.exists ? (refVm.selected_asset_info.stale ? "stale" : "current") : "unknown"
                        }

                        Label {
                            text: "Source: " + (refVm.selected_asset_info.source || "imported")
                            color: "#C7C7CC"
                            font.pixelSize: 12
                            Layout.fillWidth: true
                        }

                        Label {
                            text: "Scenes using: " + (refVm ? refVm.scenes_for_asset().join(", ") || "none" : "none")
                            color: "#8E8E93"
                            font.pixelSize: 11
                            Layout.fillWidth: true
                        }

                        RowLayout {
                            spacing: 8
                            Layout.fillWidth: true

                            Button {
                                text: "Regenerate"
                                display: AbstractButton.TextOnly
                                palette.buttonText: "#A6A6AB"
                                onClicked: {
                                    page.refVm.queue_generation("rerender", {
                                        reference_ids: [refVm.selected_asset],
                                    })
                                }
                            }

                            Button {
                                text: "MSR"
                                display: AbstractButton.TextOnly
                                palette.buttonText: "#A6A6AB"
                                onClicked: {
                                    page.refVm.queue_generation("msr_sheet", {
                                        reference_ids: [refVm.selected_asset],
                                    })
                                }
                            }
                        }
                    }
                }

                // Scene assignments
                Rectangle {
                    color: "#27272A"
                    border.color: "#3F3F46"
                    radius: 10
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 3
                        spacing: 0

                        Label {
                            text: "Scene Assignments"
                            color: "#F4F4F5"
                            font.bold: true
                            font.pixelSize: 14
                            Layout.fillWidth: true
                            Layout.topMargin: 8
                            Layout.leftMargin: 12
                        }

                        ListView {
                            id: assignmentList
                            objectName: "assignmentListView"
                            model: page.refVm ? page.refVm.assignments_model : null
                            Layout.fillWidth: true
                            Layout.fillHeight: true

                            ScrollBar.vertical: ScrollBar {}

                            delegate: Rectangle {
                                width: assignmentList.width - 6
                                height: assignmentRow.implicitHeight + 16
                                color: "transparent"
                                radius: 6

                                RowLayout {
                                    id: assignmentRow
                                    anchors.fill: parent
                                    anchors.margins: 12
                                    spacing: 12

                                    Label {
                                        text: "Scene " + model.sceneNumber
                                        color: "#F4F4F5"
                                        font.bold: true
                                        font.pixelSize: 13
                                    }

                                    Label {
                                        text: "Actors: " + model.actorIds.join(", ")
                                        color: "#8E8E93"
                                        font.pixelSize: 11
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }

                                    Label {
                                        text: "Locations: " + model.locationIds.join(", ")
                                        color: "#8E8E93"
                                        font.pixelSize: 11
                                        elide: Text.ElideRight
                                    }

                                    Label {
                                        text: "Backgrounds: " + model.backgroundIds.join(", ")
                                        color: "#8E8E93"
                                        font.pixelSize: 11
                                        elide: Text.ElideRight
                                    }
                                }
                            }

                        }
                    }
                }
            }
        }

        // Save bar
        Rectangle {
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.preferredHeight: 48

            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 10

                Label {
                    text: "Revision: " + (refVm ? refVm.revision : "")
                    color: "#8E8E93"
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }

                Button {
                    text: "Save Assignments"
                    objectName: "saveAssignmentsBtn"
                    palette.buttonText: "#F4F4F5"
                    background: Rectangle {
                        color: "#0F7253"
                        radius: 6
                    }
                    onClicked: {
                        if (page.refVm) {
                            page.refVm.save_assignments(page.refVm.collect_assignments())
                        }
                    }
                }
            }
        }
    }
}
