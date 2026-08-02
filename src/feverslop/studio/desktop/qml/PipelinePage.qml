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
        Label { text: "Pipeline action"; color: "#1C1C1E"; font.bold: true; font.pixelSize: 16 }
        RowLayout {
            Layout.fillWidth: true
            ComboBox {
                id: action
                Layout.fillWidth: true
                implicitHeight: 44
                textRole: "label"
                valueRole: "value"
                contentItem: Label {
                    objectName: "pipelineActionCurrentValue"
                    leftPadding: 12
                    rightPadding: action.indicator.width + 12
                    text: action.displayText
                    color: "#F4F4F5"
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
                background: Rectangle {
                    color: "#303033"
                    border.color: action.activeFocus ? "#5B5FC7" : "#68686D"
                    radius: 4
                }
                model: page.movie ? [
                    { label: "Movie full auto", value: "movie-full-auto" },
                    { label: "Movie references", value: "movie-references" },
                    { label: "Render selected scenes", value: "movie-render" },
                    { label: "Final concat", value: "movie-final-concat" }
                ] : (vm ? (vm.jobs, vm.pipeline_actions(page.selectedScenes)) : [])
                onModelChanged: {
                    if (page.movie) return
                    for (var i = 0; i < count; ++i) {
                        if (model[i].recommended) {
                            currentIndex = i
                            return
                        }
                    }
                }
                delegate: ItemDelegate {
                    id: delegateItem
                    required property var modelData
                    width: action.width
                    implicitHeight: 40
                    enabled: modelData.enabled !== false
                    text: modelData.recommended ? modelData.label + " (next)" : modelData.label
                    contentItem: Label {
                        text: delegateItem.text
                        color: delegateItem.enabled ? "#1C1C1E" : "#6E6E73"
                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideRight
                    }
                    background: Rectangle {
                        color: delegateItem.highlighted ? "#E8E8FF" : "#FFFFFF"
                    }
                    ToolTip.visible: hovered && !enabled && modelData.reason
                    ToolTip.text: modelData.reason || ""
                }
                popup: Popup {
                    y: action.height - 1
                    width: action.width
                    implicitHeight: contentItem.implicitHeight + topPadding + bottomPadding
                    padding: 1
                    contentItem: ListView {
                        objectName: "pipelineActionList"
                        clip: true
                        implicitHeight: contentHeight
                        model: action.popup.visible ? action.delegateModel : null
                        currentIndex: action.highlightedIndex
                        ScrollIndicator.vertical: ScrollIndicator { }
                    }
                    background: Rectangle {
                        color: "#FFFFFF"
                        border.color: "#68686D"
                        radius: 4
                    }
                }
            }
            StyledTextField {
                id: scenes
                Layout.preferredWidth: 240
                implicitHeight: 44
                placeholderText: "Scenes: 1,3,5,10"
            }
            Button {
                text: "Start"
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
            color: "#FFFFFF"
            border.color: "#D8D8DC"
            radius: 6
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                Label {
                    text: vm && vm.active_job.id ? vm.active_job.action
                         : vm && vm.jobs.length && vm.jobs[0].status === "failed" ? "Pipeline failed"
                         : "Pipeline idle"
                    color: vm && vm.jobs.length && vm.jobs[0].status === "failed" ? "#C62828" : "#1C1C1E"
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
