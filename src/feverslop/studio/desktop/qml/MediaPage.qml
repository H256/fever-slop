import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtMultimedia

Item {
    id: page
    required property bool reviewMode
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    property string selectedPath: ""

    MediaPlayer {
        id: player
        source: vm && page.selectedPath ? vm.media_url(page.selectedPath) : ""
        audioOutput: AudioOutput {}
        videoOutput: video
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 14
        Rectangle {
            Layout.preferredWidth: 320
            Layout.fillHeight: true
            color: "#FFFFFF"
            border.color: "#D8D8DC"
            radius: 6
            ListView {
                anchors.fill: parent
                anchors.margins: 10
                clip: true
                model: vm ? vm.artifact_entries : []
                delegate: ItemDelegate {
                    required property var modelData
                    visible: modelData.kind === "video" && (page.reviewMode || modelData.path.toLowerCase().includes("final"))
                    height: visible ? 50 : 0
                    width: ListView.view.width
                    text: modelData.path
                    icon.name: "video-x-generic"
                    onClicked: page.selectedPath = modelData.path
                }
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10
            Rectangle {
                color: "#111113"
                radius: 6
                Layout.fillWidth: true
                Layout.fillHeight: true
                VideoOutput { id: video; anchors.fill: parent; fillMode: VideoOutput.PreserveAspectFit }
                Label { visible: !page.selectedPath; anchors.centerIn: parent; text: "Select a video"; color: "#8E8E93" }
            }
            RowLayout {
                Layout.fillWidth: true
                ToolButton { icon.name: player.playbackState === MediaPlayer.PlayingState ? "media-playback-pause" : "media-playback-start"; ToolTip.visible: hovered; ToolTip.text: "Play or pause"; onClicked: player.playbackState === MediaPlayer.PlayingState ? player.pause() : player.play() }
                Slider { from: 0; to: Math.max(1, player.duration); value: player.position; Layout.fillWidth: true; onMoved: player.position = value }
                Label { text: Math.floor(player.position / 1000) + " / " + Math.floor(player.duration / 1000) + " s"; color: "#1C1C1E"; Layout.preferredWidth: 100 }
            }
            RowLayout {
                visible: page.reviewMode
                Layout.fillWidth: true
                Label { text: "In"; color: "#6E6E73" }
                StyledTextField {
                    id: trimIn
                    text: "0"
                    validator: DoubleValidator { bottom: 0 }
                    Layout.preferredWidth: 90
                }
                Label { text: "Out"; color: "#6E6E73" }
                StyledTextField {
                    id: trimOut
                    text: player.duration > 0 ? (player.duration / 1000).toFixed(3) : "0"
                    validator: DoubleValidator { bottom: 0 }
                    Layout.preferredWidth: 90
                }
                CheckBox { id: exact; text: "Exact re-encode" }
                Item { Layout.fillWidth: true }
                Button {
                    text: "Create recut"
                    icon.name: "document-save"
                    implicitHeight: 44
                    enabled: page.selectedPath.length > 0
                    onClicked: if (vm) vm.start_recut(page.selectedPath, page.selectedPath.replace(".mp4", "_recut.mp4"), parseFloat(trimIn.text), parseFloat(trimOut.text), exact.checked)
                }
            }
        }
    }
}
