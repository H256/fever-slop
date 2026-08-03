import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: page
    clip: true
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null

    ColumnLayout {
        x: 28
        width: Math.max(0, page.availableWidth - 56)
        spacing: 18
        Label {
            text: vm && vm.current_project_id ? vm.current_project.name : "No project selected"
            color: theme.primaryText
            font.bold: true
            font.pixelSize: 22
        }
        Label {
            text: vm && vm.current_project_id ? vm.current_project.path : ""
            color: theme.secondaryText
            font.pixelSize: 13
            elide: Text.ElideMiddle
            Layout.fillWidth: true
        }
        Flow {
            Layout.fillWidth: true
            spacing: 12
            Repeater {
                model: [
                    { key: "config", label: "Configuration" },
                    { key: "render_plan", label: "Render plan" },
                    { key: "references", label: "References" },
                    { key: "videos", label: "Video output" }
                ]
                delegate: Rectangle {
                    required property var modelData
                    width: 210
                    height: 92
                    color: theme.cardBg
                    border.color: theme.cardBorder
                    radius: 6
                    Column {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 8
                        Label { text: modelData.label; color: theme.primaryText; font.bold: true }
                        Label {
                            property string state: vm && vm.current_project.status ? (vm.current_project.status[modelData.key] || "missing") : "missing"
                            text: state === "present" ? "Ready" : "Missing"
                            color: state === "present" ? "#2E7D32" : "#B26A00"
                        }
                    }
                }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 110
            color: theme.cardBg
            border.color: theme.cardBorder
            radius: 6
            RowLayout {
                anchors.fill: parent
                anchors.margins: 18
                ColumnLayout {
                    Layout.fillWidth: true
                    Label { text: "Project type"; color: theme.secondaryText }
                    Label { text: vm ? String(vm.current_project.project_type || "standard_music_video").replace(/_/g, " ") : ""; color: theme.primaryText; font.bold: true }
                }
                ColumnLayout {
                    Label { text: "Artifact size"; color: theme.secondaryText }
                    Label {
                        text: vm && vm.current_project.artifact_sizes ? Math.round(vm.current_project.artifact_sizes.total_bytes / 1048576) + " MB" : "0 MB"
                        color: theme.primaryText
                        font.bold: true
                    }
                }
            }
        }
        Item { Layout.fillHeight: true }
    }
}
