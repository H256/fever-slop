import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

Rectangle {
    id: page
    color: "#18181B"
    property int contentPadding: 24
    property int sceneCardHeight: 116
    property var sceneVm: typeof sceneWorkspaceViewModel !== "undefined" ? sceneWorkspaceViewModel : null
    property var studioVm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    readonly property int selectedCount: sceneVm && sceneVm.selected_scene_numbers
        ? sceneVm.selected_scene_numbers.length : 0
    readonly property bool jobActive: {
        if (!studioVm || !studioVm.active_job)
            return false
        const status = studioVm.active_job.status || ""
        return status === "active" || status === "queued" || status === "running"
    }
    readonly property bool actionsEnabled: selectedCount > 0 && !jobActive
        && sceneVm && !sceneVm.submitting

    objectName: "sceneWorkspacePage"

    Component.onCompleted: if (sceneVm) sceneVm.reload()

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: page.contentPadding
        spacing: 16

        Label {
            objectName: "sceneWorkspaceErrorBanner"
            visible: page.sceneVm && !!page.sceneVm.error
            text: page.sceneVm ? page.sceneVm.error : ""
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

        Rectangle {
            visible: page.sceneVm && page.sceneVm.conflict
            color: "#FFF1E6"
            border.color: "#E07A28"
            radius: 6
            Layout.fillWidth: true
            Layout.preferredHeight: conflictRow.implicitHeight + 18

            RowLayout {
                id: conflictRow
                anchors.fill: parent
                anchors.margins: 9
                Label {
                    text: "Save conflict: reload the disk version or restore the last confirmed local view."
                    color: "#7A3E00"
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                }
                Button {
                    objectName: "reloadSceneConflictButton"
                    text: "Reload from disk"
                    Accessible.name: "Reload scene data from disk"
                    onClicked: page.sceneVm.reload()
                }
                Button {
                    objectName: "discardSceneConflictButton"
                    text: "Discard local edits"
                    Accessible.name: "Discard local scene edits"
                    onClicked: page.sceneVm.discardLocalEdits()
                }
            }
        }

        Rectangle {
            objectName: "sceneWorkspaceToolbar"
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.preferredHeight: 68

            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 12

                TextField {
                    id: filterField
                    objectName: "sceneFilter"
                    placeholderText: "Filter scene, status, or performance"
                    color: "#F4F4F5"
                    placeholderTextColor: "#A1A1AA"
                    selectionColor: "#6366F1"
                    selectedTextColor: "#FFFFFF"
                    leftPadding: 14
                    rightPadding: 14
                    Accessible.name: "Filter scenes"
                    Layout.preferredWidth: 340
                    Layout.preferredHeight: 42
                    background: Rectangle {
                        color: "#18181B"
                        border.color: filterField.activeFocus ? "#818CF8" : "#52525B"
                        radius: 7
                    }
                }
                Label {
                    objectName: "selectedSceneCount"
                    text: page.selectedCount
                        + (page.selectedCount === 1 ? " scene selected" : " scenes selected")
                        + " · Ctrl+click for multiple"
                    color: "#D4D4D8"
                    Layout.fillWidth: true
                }
                Button {
                    objectName: "discardDirtySceneButton"
                    text: "Discard local edits"
                    palette.button: "#3F3F46"
                    palette.buttonText: "#F4F4F5"
                    Accessible.name: "Discard local scene edits"
                    visible: page.sceneVm && page.sceneVm.dirty && !page.sceneVm.conflict
                    onClicked: page.sceneVm.discardLocalEdits()
                }
                Button {
                    objectName: "renderSelectedScenesButton"
                    text: "Render"
                    palette.button: "#4F46E5"
                    palette.buttonText: "#FFFFFF"
                    Accessible.name: "Render selected scenes"
                    enabled: page.actionsEnabled
                    onClicked: page.sceneVm.startSelectedAction("render")
                }
                Button {
                    objectName: "rerenderSelectedScenesButton"
                    text: "Rerender"
                    palette.button: "#3F3F46"
                    palette.buttonText: "#F4F4F5"
                    Accessible.name: "Rerender selected scenes"
                    enabled: page.actionsEnabled
                    onClicked: page.sceneVm.startSelectedAction("rerender")
                }
                Button {
                    objectName: "retakeSelectedScenesButton"
                    text: "Retake"
                    palette.button: "#3F3F46"
                    palette.buttonText: "#F4F4F5"
                    Accessible.name: "Retake selected scenes"
                    enabled: page.actionsEnabled
                    onClicked: page.sceneVm.startSelectedAction("retake")
                }
                Button {
                    objectName: "rebuildPreviewButton"
                    text: "Rebuild Preview"
                    palette.button: "#3F3F46"
                    palette.buttonText: "#F4F4F5"
                    Accessible.name: "Preview rebuild impact"
                    enabled: page.actionsEnabled
                    onClicked: rebuildDialog.open()
                }
            }
        }

        SplitView {
            orientation: Qt.Horizontal
            Layout.fillWidth: true
            Layout.fillHeight: true
            handle: Rectangle {
                implicitWidth: 10
                color: "#18181B"
                Rectangle {
                    anchors.centerIn: parent
                    width: 2
                    height: parent.height
                    color: "#52525B"
                }
            }

            ScrollView {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 520
                clip: true
                background: Rectangle { color: "#18181B" }
                ScrollBar.vertical: ScrollBar {
                    objectName: "sceneListScrollBar"
                    policy: ScrollBar.AsNeeded
                    contentItem: Rectangle {
                        objectName: "sceneListScrollThumb"
                        implicitWidth: 8
                        radius: 4
                        color: "#52525B"
                    }
                    background: Rectangle { color: "#18181B" }
                }

                ListView {
                    id: sceneList
                    objectName: "sceneCardList"
                    spacing: 12
                    model: page.sceneVm ? page.sceneVm.scenes : null
                    boundsBehavior: Flickable.StopAtBounds
                    keyNavigationEnabled: true
                    activeFocusOnTab: true
                    function activateCurrentScene() {
                        if (currentItem && currentItem.visible)
                            page.sceneVm.toggleSelection(currentItem.sceneNumber)
                    }
                    Keys.onSpacePressed: activateCurrentScene()
                    Keys.onReturnPressed: activateCurrentScene()
                    Keys.onEnterPressed: activateCurrentScene()

                    delegate: Item {
                        required property int index
                        required property int sceneNumber
                        required property real startSeconds
                        required property real endSeconds
                        required property string performanceState
                        required property string status
                        required property url thumbnailUrl
                        required property bool selected
                        readonly property string filterText: filterField.text.trim().toLowerCase()
                        readonly property bool matchesFilter: !filterText
                            || String(sceneNumber).indexOf(filterText) >= 0
                            || status.toLowerCase().indexOf(filterText) >= 0
                            || performanceState.toLowerCase().indexOf(filterText) >= 0
                        readonly property bool keyboardCurrent: ListView.isCurrentItem && sceneList.activeFocus

                        width: sceneList.width
                        height: matchesFilter ? page.sceneCardHeight : 0
                        visible: matchesFilter

                        SceneCard {
                            anchors.fill: parent
                            sceneNumber: parent.sceneNumber
                            startSeconds: parent.startSeconds
                            endSeconds: parent.endSeconds
                            performanceState: parent.performanceState
                            status: parent.status
                            thumbnailUrl: parent.thumbnailUrl
                            selected: parent.selected
                            keyboardCurrent: parent.keyboardCurrent
                            onActivated: (number, modifiers) => {
                                sceneList.currentIndex = parent.index
                                sceneList.forceActiveFocus()
                                page.sceneVm.selectScene(
                                    number,
                                    !!(modifiers & Qt.ControlModifier)
                                )
                            }
                        }
                    }

                    Label {
                        anchors.centerIn: parent
                        visible: sceneList.count === 0
                        text: "No scenes available"
                        color: "#A1A1AA"
                    }
                }
            }

            Rectangle {
                SplitView.preferredWidth: 560
                SplitView.minimumWidth: 440
                color: "#202024"
                border.color: "#3F3F46"
                radius: 10

                SceneInspector {
                    anchors.fill: parent
                    anchors.margins: 20
                    sceneVm: page.sceneVm
                }
            }
        }

        PromptRevisionDrawer {
            id: revisionDrawer
            edge: drawer.Right
            sceneVm: page.sceneVm
            rebuildVm: typeof rebuildViewModel !== "undefined" ? rebuildViewModel : null
            parent: page
        }

        RebuildPreviewDialog {
            id: rebuildDialog
            parent: page
            studioVm: page.studioVm
            sceneVm: page.sceneVm
            onRebuildAccepted: {
                const rebuildKinds = rebuildSection.artifactKinds
                if (rebuildKinds.length > 0) {
                    page.sceneVm.startSelectedAction(rebuildKinds[0])
                }
            }
        }
    }
}
