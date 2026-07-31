import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: inspector
    required property var sceneVm
    readonly property var scene: sceneVm ? sceneVm.inspectedScene : ({})

    objectName: "sceneInspector"
    clip: true
    background: Rectangle { color: "#202024" }
    ScrollBar.vertical: ScrollBar {
        policy: ScrollBar.AsNeeded
        contentItem: Rectangle {
            implicitWidth: 8
            radius: 4
            color: "#52525B"
        }
        background: Rectangle { color: "#202024" }
    }

    function loadScene() {
        shotDescription.text = scene.shotDescription || ""
        imagePrompt.text = scene.imagePrompt || ""
        const source = scene.videoPromptField || "base_prompt"
        const index = promptSource.indexOfValue(source)
        promptSource.currentIndex = index >= 0 ? index : 2
        loadPromptSource()
    }

    function loadPromptSource() {
        const prompts = scene.ltxPrompts || ({})
        videoPrompt.text = prompts[promptSource.currentValue] || ""
    }

    Component.onCompleted: loadScene()
    Connections {
        target: inspector.sceneVm
        function onInspectedSceneChanged() { inspector.loadScene() }
    }

    ColumnLayout {
        width: inspector.availableWidth
        spacing: 14

        Label {
            text: inspector.scene.sceneNumber ? "Scene " + inspector.scene.sceneNumber : "No scene selected"
            color: "#F4F4F5"
            font.pixelSize: 22
            font.bold: true
        }

        Label { text: "Shot description"; color: "#E4E4E7"; font.pixelSize: 14; font.bold: true }
        TextArea {
            id: shotDescription
            objectName: "sceneShotDescription"
            Accessible.name: "Shot description"
            enabled: !!inspector.scene.sceneNumber
            wrapMode: TextEdit.Wrap
            color: "#F4F4F5"
            placeholderTextColor: "#A1A1AA"
            selectionColor: "#6366F1"
            selectedTextColor: "#FFFFFF"
            leftPadding: 14
            rightPadding: 14
            topPadding: 12
            bottomPadding: 12
            Layout.fillWidth: true
            Layout.preferredHeight: 104
            background: Rectangle {
                color: "#18181B"
                border.color: shotDescription.activeFocus ? "#818CF8" : "#52525B"
                radius: 8
            }
        }

        RowLayout {
            spacing: 8
            Label { text: "Image prompt"; color: "#E4E4E7"; font.pixelSize: 14; font.bold: true; Layout.fillWidth: true }
            Button {
                objectName: "imagePromptHistoryButton"
                text: "↩"
                palette.button: "#3F3F46"
                palette.buttonText: "#A1A1AA"
                Accessible.name: "Show image prompt revision history"
                onClicked: {
                    if (typeof rebuildViewModel !== "undefined" && !!inspector.scene.sceneNumber) {
                        rebuildViewModel.loadRevisions(inspector.scene.sceneNumber, "z_image_prompt")
                        revisionDrawer.open(Drawer.Right)
                    }
                }
            }
        }
        TextArea {
            id: imagePrompt
            objectName: "sceneImagePrompt"
            Accessible.name: "Image prompt"
            enabled: !!inspector.scene.sceneNumber
            wrapMode: TextEdit.Wrap
            color: "#F4F4F5"
            placeholderTextColor: "#A1A1AA"
            selectionColor: "#6366F1"
            selectedTextColor: "#FFFFFF"
            leftPadding: 14
            rightPadding: 14
            topPadding: 12
            bottomPadding: 12
            Layout.fillWidth: true
            Layout.preferredHeight: 150
            background: Rectangle {
                color: "#18181B"
                border.color: imagePrompt.activeFocus ? "#818CF8" : "#52525B"
                radius: 8
            }
        }

        RowLayout {
            spacing: 8
            Label { text: "LTX prompt"; color: "#E4E4E7"; font.pixelSize: 14; font.bold: true; Layout.fillWidth: true }
            Button {
                objectName: "videoPromptHistoryButton"
                text: "↩"
                palette.button: "#3F3F46"
                palette.buttonText: "#A1A1AA"
                Accessible.name: "Show video prompt revision history"
                onClicked: {
                    if (typeof rebuildViewModel !== "undefined" && !!inspector.scene.sceneNumber) {
                        rebuildViewModel.loadRevisions(inspector.scene.sceneNumber, "i2v_prompt")
                        revisionDrawer.open(Drawer.Right)
                    }
                }
            }
            ComboBox {
                id: promptSource
                objectName: "sceneLtxPromptSource"
                Accessible.name: "LTX prompt source"
                textRole: "label"
                valueRole: "value"
                model: [
                    { label: "Original style I2V", value: "original_style_i2v_prompt" },
                    { label: "I2V from T2I", value: "i2v_prompt_from_t2i" },
                    { label: "Base prompt", value: "base_prompt" }
                ]
                onCurrentValueChanged: inspector.loadPromptSource()
                palette.text: "#F4F4F5"
                palette.buttonText: "#F4F4F5"
                palette.button: "#27272A"
            }
        }
        TextArea {
            id: videoPrompt
            objectName: "sceneLtxPrompt"
            Accessible.name: "LTX prompt"
            enabled: !!inspector.scene.sceneNumber
            wrapMode: TextEdit.Wrap
            color: "#F4F4F5"
            placeholderTextColor: "#A1A1AA"
            selectionColor: "#6366F1"
            selectedTextColor: "#FFFFFF"
            leftPadding: 14
            rightPadding: 14
            topPadding: 12
            bottomPadding: 12
            Layout.fillWidth: true
            Layout.preferredHeight: 180
            background: Rectangle {
                color: "#18181B"
                border.color: videoPrompt.activeFocus ? "#818CF8" : "#52525B"
                radius: 8
            }
        }

        Label { text: "References"; color: "#E4E4E7"; font.bold: true }
        Label {
            text: inspector.scene.referenceIds && inspector.scene.referenceIds.length
                ? inspector.scene.referenceIds.join(", ") : "None"
            color: "#A1A1AA"
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Label { text: "Output paths"; color: "#E4E4E7"; font.bold: true }
        Label {
            text: [
                inspector.scene.thumbnailPath ? "Preview: " + inspector.scene.thumbnailPath : "",
                inspector.scene.workflowPath ? "Workflow: " + inspector.scene.workflowPath : "",
                inspector.scene.videoPath ? "Video: " + inspector.scene.videoPath : ""
            ].filter(Boolean).join("\n") || "No outputs"
            color: "#A1A1AA"
            wrapMode: Text.WrapAnywhere
            font.pixelSize: 12
            Layout.fillWidth: true
        }
        Label {
            visible: !!inspector.scene.failureMessage
            text: inspector.scene.failureMessage || ""
            color: "#F87171"
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Button {
            objectName: "saveScenePromptsButton"
            text: "Save prompt fields"
            palette.button: "#4F46E5"
            palette.buttonText: "#FFFFFF"
            Accessible.name: "Save prompt fields for selected scene"
            enabled: !!inspector.scene.sceneNumber
            Layout.alignment: Qt.AlignRight
            onClicked: inspector.sceneVm.savePromptFields(
                inspector.scene.sceneNumber,
                {
                    "shot_description": shotDescription.text,
                    "image_prompt": imagePrompt.text,
                    "video_prompt": videoPrompt.text
                },
                promptSource.currentValue
            )
        }
    }
}
