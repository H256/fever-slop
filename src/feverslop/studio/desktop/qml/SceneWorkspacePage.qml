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

    objectName: "sceneWorkspacePage"

    Component.onCompleted: if (sceneVm) sceneVm.reload()

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 10

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
                    text: "Save conflict: the project changed on disk. Both choices reload the server copy."
                    color: "#7A3E00"
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                }
                Button {
                    text: "Reload"
                    Accessible.name: "Reload scene data from disk"
                    onClicked: page.sceneVm.reload()
                }
                Button {
                    text: "Discard local edits"
                    Accessible.name: "Discard local scene edits and reload"
                    onClicked: page.sceneVm.reload()
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

                    delegate: Item {
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
                            onActivated: number => page.sceneVm.toggleSelection(number)
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
