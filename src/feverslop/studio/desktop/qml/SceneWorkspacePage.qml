import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

Item {
    id: page
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
        anchors.margins: 18
        spacing: 10

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
            color: "#FFFFFF"
            border.color: "#D8D8DC"
            radius: 6
            Layout.fillWidth: true
            Layout.preferredHeight: 54

            RowLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8

                TextField {
                    id: filterField
                    objectName: "sceneFilter"
                    placeholderText: "Filter scene, status, or performance"
                    Accessible.name: "Filter scenes"
                    Layout.preferredWidth: 280
                }
                Label {
                    objectName: "selectedSceneCount"
                    text: page.selectedCount + (page.selectedCount === 1 ? " scene selected" : " scenes selected")
                    color: "#48484A"
                    Layout.fillWidth: true
                }
                Button {
                    objectName: "discardDirtySceneButton"
                    text: "Discard local edits"
                    Accessible.name: "Discard local scene edits"
                    visible: page.sceneVm && page.sceneVm.dirty && !page.sceneVm.conflict
                    onClicked: page.sceneVm.discardLocalEdits()
                }
                Button {
                    objectName: "renderSelectedScenesButton"
                    text: "Render"
                    Accessible.name: "Render selected scenes"
                    enabled: page.actionsEnabled
                    onClicked: page.sceneVm.startSelectedAction("render")
                }
                Button {
                    objectName: "rerenderSelectedScenesButton"
                    text: "Rerender"
                    Accessible.name: "Rerender selected scenes"
                    enabled: page.actionsEnabled
                    onClicked: page.sceneVm.startSelectedAction("rerender")
                }
                Button {
                    objectName: "retakeSelectedScenesButton"
                    text: "Retake"
                    Accessible.name: "Retake selected scenes"
                    enabled: page.actionsEnabled
                    onClicked: page.sceneVm.startSelectedAction("retake")
                }
            }
        }

        SplitView {
            orientation: Qt.Horizontal
            Layout.fillWidth: true
            Layout.fillHeight: true

            ScrollView {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 360
                clip: true

                ListView {
                    id: sceneList
                    objectName: "sceneCardList"
                    spacing: 7
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
                        height: matchesFilter ? 76 : 0
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
                            onActivated: number => {
                                sceneList.currentIndex = parent.index
                                sceneList.forceActiveFocus()
                                page.sceneVm.toggleSelection(number)
                            }
                        }
                    }

                    Label {
                        anchors.centerIn: parent
                        visible: sceneList.count === 0
                        text: "No scenes available"
                        color: "#6E6E73"
                    }
                }
            }

            Rectangle {
                SplitView.preferredWidth: 440
                SplitView.minimumWidth: 330
                color: "#FFFFFF"
                border.color: "#D8D8DC"

                SceneInspector {
                    anchors.fill: parent
                    anchors.margins: 14
                    sceneVm: page.sceneVm
                }
            }
        }
    }
}
