import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Item {
    id: page
    required property string heading
    required property string defaultPath
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            TextField { id: pathField; text: page.defaultPath; placeholderText: "Relative JSON path"; Layout.fillWidth: true; implicitHeight: 44 }
            Button {
                visible: page.defaultPath === "config.json"
                text: "Import Audio"
                icon.name: "audio-x-generic"
                implicitHeight: 44
                onClicked: audioDialog.open()
            }
            Button { text: "Open"; icon.name: "document-open"; implicitHeight: 44; onClicked: if (vm) vm.load_json_artifact(pathField.text) }
            Button {
                text: "Save"
                icon.name: "document-save"
                implicitHeight: 44
                palette.buttonText: "#FFFFFF"
                background: Rectangle { color: parent.hovered ? "#666AD1" : "#5B5FC7"; radius: 6 }
                onClicked: if (vm) vm.save_json_artifact(pathField.text, editor.text)
            }
        }
        Rectangle {
            color: "#FFFFFF"
            border.color: "#D8D8DC"
            radius: 6
            Layout.fillWidth: true
            Layout.fillHeight: true
            ScrollView {
                anchors.fill: parent
                anchors.margins: 1
                TextArea {
                    id: editor
                    text: vm ? vm.editor_text : ""
                    color: "#1C1C1E"
                    selectionColor: "#7B83EB"
                    selectedTextColor: "#FFFFFF"
                    font.family: "monospace"
                    font.pixelSize: 13
                    wrapMode: TextEdit.NoWrap
                    leftPadding: 18
                    rightPadding: 18
                    topPadding: 16
                    bottomPadding: 16
                    background: null
                }
            }
        }
    }

    FileDialog {
        id: audioDialog
        title: "Select project audio"
        nameFilters: ["Audio files (*.mp3 *.wav *.flac *.m4a *.ogg)"]
        onAccepted: if (vm) vm.import_audio(selectedFile)
    }
}
