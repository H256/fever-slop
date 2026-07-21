import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: inspector
    required property var sceneVm
    readonly property var scene: sceneVm ? sceneVm.inspectedScene : ({})

    objectName: "sceneInspector"
    clip: true

    function loadScene() {
        shotDescription.text = scene.shotDescription || ""
        imagePrompt.text = scene.imagePrompt || ""
        videoPrompt.text = scene.videoPrompt || ""
        const source = scene.videoPromptField || "base_prompt"
        const index = promptSource.indexOfValue(source)
        promptSource.currentIndex = index >= 0 ? index : 2
    }

    Component.onCompleted: loadScene()
    Connections {
        target: inspector.sceneVm
        function onInspectedSceneChanged() { inspector.loadScene() }
    }

    ColumnLayout {
        width: inspector.availableWidth
        spacing: 10

        Label {
            text: inspector.scene.sceneNumber ? "Scene " + inspector.scene.sceneNumber : "No scene selected"
            color: "#1C1C1E"
            font.pixelSize: 18
            font.bold: true
        }

        Label { text: "Shot description"; color: "#48484A"; font.bold: true }
        TextArea {
            id: shotDescription
            objectName: "sceneShotDescription"
            Accessible.name: "Shot description"
            enabled: !!inspector.scene.sceneNumber
            wrapMode: TextEdit.Wrap
            Layout.fillWidth: true
            Layout.preferredHeight: 84
        }

        Label { text: "Image prompt"; color: "#48484A"; font.bold: true }
        TextArea {
            id: imagePrompt
            objectName: "sceneImagePrompt"
            Accessible.name: "Image prompt"
            enabled: !!inspector.scene.sceneNumber
            wrapMode: TextEdit.Wrap
            Layout.fillWidth: true
            Layout.preferredHeight: 110
        }

        RowLayout {
            Label { text: "LTX prompt"; color: "#48484A"; font.bold: true; Layout.fillWidth: true }
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
            }
        }
        TextArea {
            id: videoPrompt
            objectName: "sceneLtxPrompt"
            Accessible.name: "LTX prompt"
            enabled: !!inspector.scene.sceneNumber
            wrapMode: TextEdit.Wrap
            Layout.fillWidth: true
            Layout.preferredHeight: 130
        }

        Label { text: "References"; color: "#48484A"; font.bold: true }
        Label {
            text: inspector.scene.referenceIds && inspector.scene.referenceIds.length
                ? inspector.scene.referenceIds.join(", ") : "None"
            color: "#6E6E73"
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Label { text: "Output paths"; color: "#48484A"; font.bold: true }
        Label {
            text: [
                inspector.scene.thumbnailPath ? "Preview: " + inspector.scene.thumbnailPath : "",
                inspector.scene.workflowPath ? "Workflow: " + inspector.scene.workflowPath : "",
                inspector.scene.videoPath ? "Video: " + inspector.scene.videoPath : ""
            ].filter(Boolean).join("\n") || "No outputs"
            color: "#6E6E73"
            wrapMode: Text.WrapAnywhere
            font.pixelSize: 12
            Layout.fillWidth: true
        }
        Label {
            visible: !!inspector.scene.failureMessage
            text: inspector.scene.failureMessage || ""
            color: "#C62828"
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Button {
            objectName: "saveScenePromptsButton"
            text: "Save prompt fields"
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
