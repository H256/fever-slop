import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Item {
    id: page
    objectName: renderPlanMode ? "renderPlanWorkspace" : "configWorkspace"
    required property string heading
    required property string defaultPath
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    property bool renderPlanMode: defaultPath !== "config.json"
    property bool rawMode: !renderPlanMode
    property var selectedScene: null

    function choosePreferredPath() {
        if (!vm || !renderPlanMode) return
        var preferred = vm.preferred_artifact("render_plans")
        if (preferred) {
            pathField.text = preferred
            vm.load_json_artifact(preferred)
        }
    }

    Component.onCompleted: choosePreferredPath()
    Connections {
        target: vm
        function onCurrentProjectChanged() { page.choosePreferredPath() }
        function onEditorChanged() {
            page.selectedScene = null
            editor.applyModelText()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            StyledTextField { id: pathField; text: page.defaultPath; placeholderText: "Relative JSON path"; Layout.fillWidth: true; implicitHeight: 44 }
            ButtonGroup { id: editorMode }
            ToolButton {
                visible: page.renderPlanMode
                text: "Inspector"
                checkable: true
                checked: !page.rawMode
                ButtonGroup.group: editorMode
                onClicked: page.rawMode = false
                ToolTip.visible: hovered
                ToolTip.text: "Structured scene inspector"
            }
            ToolButton {
                visible: page.renderPlanMode
                text: "JSON"
                checkable: true
                checked: page.rawMode
                ButtonGroup.group: editorMode
                onClicked: page.rawMode = true
                ToolTip.visible: hovered
                ToolTip.text: "Raw JSON editor"
            }
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
                visible: !page.renderPlanMode || page.rawMode
                implicitHeight: 44
                palette.buttonText: "#FFFFFF"
                background: Rectangle { color: parent.hovered ? "#666AD1" : "#5B5FC7"; radius: 6 }
                onClicked: if (vm) vm.save_json_artifact(pathField.text, editor.text)
            }
        }
        StackLayout {
            currentIndex: page.rawMode ? 1 : 0
            Layout.fillWidth: true
            Layout.fillHeight: true

            RowLayout {
                spacing: 12
                Rectangle {
                    Layout.preferredWidth: 250
                    Layout.fillHeight: true
                    color: theme.cardBg
                    border.color: theme.cardBorder
                    radius: 6
                    ListView {
                        anchors.fill: parent
                        anchors.margins: 8
                        clip: true
                        spacing: 4
                        model: vm ? vm.editor_scenes : []
                        delegate: ItemDelegate {
                            required property var modelData
                            width: ListView.view.width
                            height: 52
                            text: "Scene " + modelData.scene
                            highlighted: page.selectedScene && page.selectedScene.scene === modelData.scene
                            onClicked: {
                                page.selectedScene = modelData
                                promptField.text = modelData.prompt || modelData.target_prompt || ""
                                shotField.text = modelData.shot_description || modelData.description || ""
                            }
                        }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: theme.cardBg
                    border.color: theme.cardBorder
                    radius: 6
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 10
                        Label {
                            text: page.selectedScene ? "Scene " + page.selectedScene.scene : "Select a scene"
                            color: theme.primaryText
                            font.bold: true
                            font.pixelSize: 18
                        }
                        Label { text: "Target prompt"; color: theme.secondaryText }
                        ScrollView {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            StyledTextArea {
                                id: promptField
                                wrapMode: TextEdit.Wrap
                                color: theme.primaryText
                                background: Rectangle { color: theme.inputBg; border.color: theme.inputBorder; radius: 4 }
                            }
                        }
                        Label { text: "Shot description"; color: theme.secondaryText }
                        StyledTextArea {
                            id: shotField
                            Layout.fillWidth: true
                            Layout.preferredHeight: 100
                            wrapMode: TextEdit.Wrap
                            color: theme.primaryText
                            background: Rectangle { color: theme.inputBg; border.color: theme.inputBorder; radius: 4 }
                        }
                        Button {
                            text: "Save scene"
                            icon.name: "document-save"
                            enabled: page.selectedScene !== null
                            implicitHeight: 44
                            Layout.alignment: Qt.AlignRight
                            palette.buttonText: "#FFFFFF"
                            background: Rectangle { color: parent.enabled ? (parent.hovered ? "#666AD1" : "#5B5FC7") : "#AEAEB2"; radius: 6 }
                            onClicked: {
                                if (!vm || !page.selectedScene) return
                                var updates = { prompt: promptField.text, shot_description: shotField.text }
                                if (vm.patch_render_scene(pathField.text, page.selectedScene.scene, updates)) page.selectedScene = null
                            }
                        }
                    }
                }
            }

            Rectangle {
                color: theme.cardBg
                border.color: theme.cardBorder
                radius: 6
                ScrollView {
                    anchors.fill: parent
                    anchors.margins: 1
                    StyledTextArea {
                        id: editor
                        objectName: page.renderPlanMode ? "renderPlanJsonEditor" : "configJsonEditor"
                        property bool applyingModelText: false
                        function applyModelText() {
                            var modelText = vm ? vm.editor_text : ""
                            if (text === modelText) return
                            applyingModelText = true
                            text = modelText
                            applyingModelText = false
                        }
                        Component.onCompleted: applyModelText()
                        onTextChanged: {
                            if (!applyingModelText && vm) vm.set_json_editor_draft(text)
                        }
                        color: theme.primaryText
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
    }

    FileDialog {
        id: audioDialog
        title: "Select project audio"
        nameFilters: ["Audio files (*.mp3 *.wav *.flac *.m4a *.ogg)"]
        onAccepted: if (vm) vm.import_audio(selectedFile)
    }
}
