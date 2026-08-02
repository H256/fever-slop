import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

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
                property bool expanded: false
                readonly property bool failed: modelData.status === "failed"
                readonly property string failureText: String(modelData.error || "No error details available")
                readonly property string logText: (modelData.logs || modelData.recent_logs || []).join("\n")
                width: ListView.view.width
                height: expanded && failed ? 250 : 86
                color: "#FFFFFF"
                border.color: "#D8D8DC"
                radius: 6
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    RowLayout {
                        Layout.fillWidth: true
                        ColumnLayout {
                            Layout.fillWidth: true
                            Label { text: modelData.action; color: "#1C1C1E"; font.bold: true }
                            Label { text: modelData.current_step || modelData.id; color: "#6E6E73"; elide: Text.ElideRight; Layout.fillWidth: true }
                        }
                        Label { text: modelData.status; color: failed ? "#C62828" : modelData.status === "succeeded" ? "#2E7D32" : "#5B5FC7"; font.bold: true }
                        ArtifactFreshnessBadge {
                            status: modelData.status === "succeeded" ? "current" : failed ? "stale" : "unknown"
                        }
                        ProgressBar { from: 0; to: 100; value: modelData.overall_progress || modelData.progress || 0; Layout.preferredWidth: 220 }
                    }
                    Label {
                        visible: failed
                        Layout.fillWidth: true
                        text: failureText
                        color: "#C62828"
                        elide: expanded ? Text.ElideNone : Text.ElideRight
                        wrapMode: expanded ? Text.Wrap : Text.NoWrap
                    }
                    ScrollView {
                        visible: expanded && failed
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        StyledTextArea {
                            readOnly: true
                            text: logText.length > 0 ? logText : failureText
                            color: "#E5E5EA"
                            font.family: "monospace"
                            font.pixelSize: 11
                            wrapMode: TextEdit.Wrap
                            background: Rectangle { color: "#252528"; radius: 4 }
                        }
                    }
                }
                MouseArea {
                    id: detailsMouse
                    width: parent.width
                    height: 86
                    enabled: failed
                    hoverEnabled: true
                    onClicked: expanded = !expanded
                }
                ToolTip {
                    visible: detailsMouse.containsMouse && failed && !expanded
                    text: failureText
                }
            }
        }
    }
}
