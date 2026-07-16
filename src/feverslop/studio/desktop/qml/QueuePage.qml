import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Studio jobs"; color: "#1C1C1E"; font.bold: true; font.pixelSize: 18; Layout.fillWidth: true }
            Button { icon.name: "view-refresh"; text: "Refresh"; implicitHeight: 44; onClicked: if (vm) vm.refresh_jobs() }
        }
        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 8
            clip: true
            model: vm ? vm.jobs : []
            delegate: Rectangle {
                required property var modelData
                width: ListView.view.width
                height: 86
                color: "#FFFFFF"
                border.color: "#D8D8DC"
                radius: 6
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { text: modelData.action; color: "#1C1C1E"; font.bold: true }
                        Label { text: modelData.current_step || modelData.id; color: "#6E6E73"; elide: Text.ElideRight; Layout.fillWidth: true }
                    }
                    Label { text: modelData.status; color: modelData.status === "failed" ? "#C62828" : modelData.status === "succeeded" ? "#2E7D32" : "#5B5FC7"; font.bold: true }
                    ProgressBar { from: 0; to: 100; value: modelData.overall_progress || modelData.progress || 0; Layout.preferredWidth: 220 }
                }
            }
        }
    }
}
