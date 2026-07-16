import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtMultimedia

Item {
    id: page
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    property int selectedScene: 0
    property real zoom: 1
    property real scrubSeconds: 0
    property var selectedItem: {
        var items = vm ? vm.review_items : []
        for (var i = 0; i < items.length; ++i) if (items[i].scene === selectedScene) return items[i]
        return items.length ? items[0] : null
    }

    function loadTimeline() {
        if (!vm || !vm.current_project_id || !vm.load_review_timeline()) return
        if (vm.review_items.length) selectedScene = vm.review_items[0].scene
    }

    function selectAt(index) {
        if (!vm || index < 0 || index >= vm.review_items.length) return
        selectedScene = vm.review_items[index].scene
        scrubSeconds = vm.review_items[index].start
    }

    Component.onCompleted: loadTimeline()
    Connections {
        target: vm
        function onCurrentProjectChanged() { page.loadTimeline() }
    }

    MediaPlayer {
        id: player
        source: vm && page.selectedItem && page.selectedItem.clip ? vm.media_url(page.selectedItem.clip) : ""
        audioOutput: AudioOutput {}
        videoOutput: videoOutput
        onPositionChanged: {
            if (page.selectedItem) page.scrubSeconds = page.selectedItem.start + position / 1000
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            ToolButton {
                text: "Undo"
                icon.name: "edit-undo"
                display: AbstractButton.TextBesideIcon
                enabled: vm && vm.review_can_undo
                implicitWidth: 78
                implicitHeight: 40
                ToolTip.visible: hovered
                ToolTip.text: "Undo timeline edit"
                onClicked: if (vm) vm.undo_review_timeline()
            }
            ToolButton {
                text: "Redo"
                icon.name: "edit-redo"
                display: AbstractButton.TextBesideIcon
                enabled: vm && vm.review_can_redo
                implicitWidth: 78
                implicitHeight: 40
                ToolTip.visible: hovered
                ToolTip.text: "Redo timeline edit"
                onClicked: if (vm) vm.redo_review_timeline()
            }
            Button {
                text: "Save timeline"
                icon.name: "document-save"
                enabled: vm && vm.review_dirty
                implicitHeight: 40
                onClicked: if (vm) vm.save_review_timeline()
            }
            Button {
                text: "Render retake"
                icon.name: "view-refresh"
                enabled: page.selectedScene > 0
                implicitHeight: 40
                onClicked: {
                    if (!vm) return
                    var action = vm.current_project.project_type === "movie" ? "movie-render" : "ltx-render-scenes"
                    vm.start_job(action, [page.selectedScene])
                }
            }
            Item { Layout.fillWidth: true }
            Label { text: "Zoom"; color: "#6E6E73" }
            Slider { from: 0.75; to: 4; value: page.zoom; onMoved: page.zoom = value; Layout.preferredWidth: 150 }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 250
            spacing: 12

            Rectangle {
                color: "#111113"
                radius: 6
                Layout.fillWidth: true
                Layout.fillHeight: true
                VideoOutput { id: videoOutput; anchors.fill: parent; fillMode: VideoOutput.PreserveAspectFit }
                Label {
                    visible: !page.selectedItem || !page.selectedItem.clip
                    anchors.centerIn: parent
                    text: "No rendered clip for this shot"
                    color: "#AEAEB2"
                }
            }

            Rectangle {
                Layout.preferredWidth: 300
                Layout.fillHeight: true
                color: "#FFFFFF"
                border.color: "#D8D8DC"
                radius: 6
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    Label { text: page.selectedItem ? "Scene " + page.selectedItem.scene : "No scene"; color: "#1C1C1E"; font.bold: true; font.pixelSize: 18 }
                    Label {
                        text: page.selectedItem ? page.selectedItem.status : "missing"
                        color: page.selectedItem && page.selectedItem.status === "final" ? "#2E7D32" : "#B26A00"
                        font.bold: true
                    }
                    Label {
                        text: page.selectedItem ? page.selectedItem.preview : ""
                        color: "#6E6E73"
                        wrapMode: Text.WordWrap
                        elide: Text.ElideRight
                        maximumLineCount: 5
                        Layout.fillWidth: true
                    }
                    Item { Layout.fillHeight: true }
                    Label { text: "Raw in"; color: "#6E6E73" }
                    TextField {
                        id: trimIn
                        text: page.selectedItem ? Number(page.selectedItem.raw_in_seconds).toFixed(3) : "0"
                        validator: DoubleValidator { bottom: 0 }
                        Layout.fillWidth: true
                    }
                    Label { text: "Raw out"; color: "#6E6E73" }
                    TextField {
                        id: trimOut
                        text: page.selectedItem ? Number(page.selectedItem.raw_out_seconds).toFixed(3) : "0"
                        validator: DoubleValidator { bottom: 0 }
                        Layout.fillWidth: true
                    }
                    CheckBox { id: exactRecut; text: "Exact re-encode"; checked: true }
                    RowLayout {
                        Layout.fillWidth: true
                        Button {
                            text: "Apply trim"
                            Layout.fillWidth: true
                            enabled: page.selectedScene > 0 && Number(trimOut.text) > Number(trimIn.text)
                            onClicked: if (vm) vm.trim_review_scene(page.selectedScene, Number(trimIn.text), Number(trimOut.text))
                        }
                        Button {
                            text: "Recut"
                            icon.name: "edit-cut"
                            Layout.fillWidth: true
                            enabled: page.selectedItem && page.selectedItem.raw_clip && Number(trimOut.text) > Number(trimIn.text)
                            onClicked: {
                                if (!vm || !page.selectedItem) return
                                vm.trim_review_scene(page.selectedScene, Number(trimIn.text), Number(trimOut.text))
                                vm.save_review_timeline()
                                var output = page.selectedItem.final_clip || page.selectedItem.raw_clip.replace("/raw/", "/final/").replace("_raw.mp4", ".mp4")
                                vm.start_recut(page.selectedItem.raw_clip, output, Number(trimIn.text), Number(trimOut.text), exactRecut.checked)
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            ToolButton {
                text: player.playbackState === MediaPlayer.PlayingState ? "Pause" : "Play"
                icon.name: player.playbackState === MediaPlayer.PlayingState ? "media-playback-pause" : "media-playback-start"
                display: AbstractButton.TextBesideIcon
                implicitWidth: 82
                implicitHeight: 40
                ToolTip.visible: hovered
                ToolTip.text: "Play or pause selected shot"
                onClicked: player.playbackState === MediaPlayer.PlayingState ? player.pause() : player.play()
            }
            Slider {
                from: 0
                to: Math.max(0.001, vm ? vm.review_duration : 0.001)
                value: page.scrubSeconds
                Layout.fillWidth: true
                onMoved: {
                    page.scrubSeconds = value
                    if (!vm) return
                    var items = vm.review_items
                    for (var i = 0; i < items.length; ++i) {
                        if (value >= items[i].start && value <= items[i].end) {
                            page.selectedScene = items[i].scene
                            player.position = Math.max(0, (value - items[i].start) * 1000)
                            break
                        }
                    }
                }
            }
            Label { text: page.scrubSeconds.toFixed(2) + " / " + (vm ? vm.review_duration.toFixed(2) : "0.00") + " s"; color: "#1C1C1E"; Layout.preferredWidth: 120 }
        }

        Rectangle {
            objectName: "reviewTimeline"
            Layout.fillWidth: true
            Layout.preferredHeight: 190
            color: "#252528"
            radius: 6

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 6
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "SHOTS"; color: "#8E8E93"; font.bold: true; font.pixelSize: 11; Layout.fillWidth: true }
                    Label { text: "Use arrows to change story order"; color: "#8E8E93"; font.pixelSize: 11 }
                }
                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    ScrollBar.vertical.policy: ScrollBar.AlwaysOff
                    Row {
                        spacing: 6
                        Repeater {
                            model: vm ? vm.review_items : []
                            delegate: Rectangle {
                                required property var modelData
                                required property int index
                                width: Math.max(130, modelData.duration * 72 * page.zoom)
                                height: 122
                                color: page.selectedScene === modelData.scene ? "#4A4EB3" : modelData.stale ? "#5A4520" : "#3A3A3C"
                                border.color: page.selectedScene === modelData.scene ? "#9DA2FF" : "#55555A"
                                radius: 5
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: "Scene " + modelData.scene; color: "#FFFFFF"; font.bold: true; Layout.fillWidth: true }
                                        Label { text: modelData.status; color: modelData.status === "final" ? "#7EDB82" : "#FFCA5C"; font.pixelSize: 11 }
                                    }
                                    Label { text: modelData.preview || modelData.clip; color: "#C7C7CC"; elide: Text.ElideRight; Layout.fillWidth: true }
                                    Item { Layout.fillHeight: true }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        ToolButton {
                                            text: "<"
                                            palette.buttonText: "#FFFFFF"
                                            palette.disabled.buttonText: "#8E8E93"
                                            enabled: index > 0
                                            implicitWidth: 36
                                            implicitHeight: 36
                                            ToolTip.visible: hovered
                                            ToolTip.text: "Move shot left"
                                            onClicked: { if (vm) vm.move_review_scene(index, index - 1); page.selectedScene = modelData.scene }
                                        }
                                        Label { text: Number(modelData.duration).toFixed(2) + "s"; color: "#E5E5EA"; horizontalAlignment: Text.AlignHCenter; Layout.fillWidth: true }
                                        ToolButton {
                                            text: ">"
                                            palette.buttonText: "#FFFFFF"
                                            palette.disabled.buttonText: "#8E8E93"
                                            enabled: vm && index < vm.review_items.length - 1
                                            implicitWidth: 36
                                            implicitHeight: 36
                                            ToolTip.visible: hovered
                                            ToolTip.text: "Move shot right"
                                            onClicked: { if (vm) vm.move_review_scene(index, index + 1); page.selectedScene = modelData.scene }
                                        }
                                    }
                                }
                                TapHandler { onTapped: page.selectAt(index) }
                            }
                        }
                    }
                }
            }
        }
    }
}
