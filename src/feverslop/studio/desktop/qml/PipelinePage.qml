import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    property bool movie: vm && vm.current_project.project_type === "movie"
    property var selectedScenes: {
        var selected = []
        var parts = scenes.text.split(",")
        for (var i = 0; i < parts.length; ++i) {
            var number = parseInt(parts[i].trim())
            if (!isNaN(number) && selected.indexOf(number) === -1) selected.push(number)
        }
        return selected
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 28
        spacing: 16
        Label { text: "Pipeline action"; color: theme.primaryText; font.bold: true; font.pixelSize: 16 }
        RowLayout {
            Layout.fillWidth: true
            z: action.expanded ? 1 : 0
            Item {
                id: action
                objectName: "pipelineActionSelector"
                Layout.fillWidth: true
                Layout.preferredHeight: expanded ? 44 + Math.min(actions.length * 40, 280) : 44
                property bool expanded: false
                property var actions: page.movie ? [
                    { label: "Movie full auto", value: "movie-full-auto" },
                    { label: "Movie references", value: "movie-references" },
                    { label: "Render selected scenes", value: "movie-render" },
                    { label: "Final concat", value: "movie-final-concat" }
                ] : (vm ? (vm.current_project_id, vm.jobs, vm.pipeline_actions(page.selectedScenes)) : [])
                property int currentIndex: -1
                readonly property var currentItem: currentIndex >= 0 && currentIndex < actions.length ? actions[currentIndex] : null
                readonly property string currentValue: currentItem ? currentItem.value : ""
                onActionsChanged: {
                    for (var i = 0; i < actions.length; ++i) {
                        if (actions[i].recommended) {
                            currentIndex = i
                            return
                        }
                    }
                    currentIndex = actions.length > 0 ? 0 : -1
                }
                Rectangle {
                    width: parent.width; height: 44; color: "#303033"; border.color: "#68686D"; radius: 4
                    Label { anchors.left: parent.left; anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter; leftPadding: 12; rightPadding: 36; text: action.currentItem ? action.currentItem.label : "No pipeline actions available"; color: "#F4F4F5"; elide: Text.ElideRight }
                    Label { anchors.right: parent.right; anchors.rightMargin: 12; anchors.verticalCenter: parent.verticalCenter; text: action.expanded ? "▲" : "▼"; color: "#C7C7CC" }
                    MouseArea { anchors.fill: parent; onClicked: action.expanded = !action.expanded }
                }
                ListView {
                    visible: action.expanded; y: 44; width: parent.width; height: Math.min(contentHeight, 280); clip: true; model: action.actions
                    delegate: Rectangle {
                        required property var modelData
                        width: action.width; height: 40; color: modelData.enabled === false ? theme.disabledItemBg : theme.cardBg
                        Label { anchors.fill: parent; anchors.leftMargin: 12; verticalAlignment: Text.AlignVCenter; text: modelData.recommended ? modelData.label + " (next)" : modelData.label; color: modelData.enabled === false ? theme.secondaryText : theme.primaryText }
                        MouseArea { anchors.fill: parent; enabled: modelData.enabled !== false; onClicked: { action.currentIndex = index; action.expanded = false } }
                    }
                }
            }
            StyledTextField {
                id: scenes
                Layout.alignment: Qt.AlignTop
                Layout.preferredWidth: 240
                implicitHeight: 44
                placeholderText: "Scenes: 1,3,5,10"
            }
            Button {
                text: "Start"
                Layout.alignment: Qt.AlignTop
                icon.name: "media-playback-start"
                implicitHeight: 44
                enabled: vm && !vm.active_job.id && action.currentIndex >= 0
                         && (!action.currentItem || action.currentItem.enabled)
                palette.buttonText: "#FFFFFF"
                background: Rectangle { color: parent.enabled ? (parent.hovered ? "#666AD1" : "#5B5FC7") : "#AEAEB2"; radius: 6 }
                onClicked: {
                    if (vm) vm.start_job(action.currentValue, page.selectedScenes)
                }
            }
        }
        Label {
            Layout.fillWidth: true
            visible: !page.movie && action.currentItem && !action.currentItem.enabled
            text: visible ? action.currentItem.reason : ""
            color: "#9C2C2C"
            wrapMode: Text.Wrap
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 88
            color: theme.cardBg
            border.color: theme.cardBorder
            radius: 6
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                Label {
                    text: vm && vm.active_job.id ? vm.active_job.action
                         : vm && vm.jobs.length && vm.jobs[0].status === "failed" ? "Pipeline failed"
                         : "Pipeline idle"
                    color: vm && vm.jobs.length && vm.jobs[0].status === "failed" ? "#C62828" : theme.primaryText
                    font.bold: true
                    Layout.fillWidth: true
                }
                ProgressBar { from: 0; to: 100; value: vm && vm.active_job.id ? (vm.active_job.overall_progress || 0) : 0; Layout.preferredWidth: 300 }
            }
        }
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            StyledTextArea {
                readOnly: true
                text: vm ? vm.job_logs : "Ready."
                color: "#E5E5EA"
                font.family: "monospace"
                font.pixelSize: 12
                background: Rectangle { color: "#252528"; radius: 6 }
                leftPadding: 16
                topPadding: 14
            }
        }
    }
}
