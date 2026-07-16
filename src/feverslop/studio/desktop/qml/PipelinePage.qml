import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    property bool movie: vm && vm.current_project.project_type === "movie"

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
                model: page.movie ? [
                    { label: "Movie full auto", value: "movie-full-auto" },
                    { label: "Movie references", value: "movie-references" },
                    { label: "Render selected scenes", value: "movie-render" },
                    { label: "Final concat", value: "movie-final-concat" }
                ] : [
                    { label: "Full pipeline", value: "full-pipeline" },
                    { label: "Main pipeline", value: "main-pipeline" },
                    { label: "MSR references", value: "msr-references" },
                    { label: "MSR enrichment", value: "msr-enrich" },
                    { label: "Render selected scenes", value: "ltx-render-scenes" },
                    { label: "Final concat", value: "final-concat" }
                ]
            }
            TextField {
                id: scenes
                Layout.preferredWidth: 240
                implicitHeight: 44
                placeholderText: "Scenes: 1,3,5,10"
            }
            Button {
                text: "Start"
                icon.name: "media-playback-start"
                implicitHeight: 44
                enabled: vm && !vm.active_job.id
                palette.buttonText: "#FFFFFF"
                background: Rectangle { color: parent.enabled ? (parent.hovered ? "#666AD1" : "#5B5FC7") : "#AEAEB2"; radius: 6 }
                onClicked: {
                    var selected = []
                    var parts = scenes.text.split(",")
                    for (var i = 0; i < parts.length; ++i) {
                        var number = parseInt(parts[i].trim())
                        if (!isNaN(number)) selected.push(number)
                    }
                    if (vm) vm.start_job(action.currentValue, selected)
                }
            }
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
                Label { text: vm && vm.active_job.id ? vm.active_job.action : "Pipeline idle"; color: "#1C1C1E"; font.bold: true; Layout.fillWidth: true }
                ProgressBar { from: 0; to: 100; value: vm && vm.active_job.id ? (vm.active_job.overall_progress || 0) : 0; Layout.preferredWidth: 300 }
            }
        }
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            TextArea {
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
