import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Item {
    id: page
    required property string categoryFilter
    required property string titleText
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    property string selectedPath: ""
    property string selectedKind: ""

    function matches(entry) {
        if (!categoryFilter) return true
        return entry.category === categoryFilter || entry.path.toLowerCase().includes(categoryFilter)
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16
        Rectangle {
            Layout.preferredWidth: 360
            Layout.fillHeight: true
            color: "#FFFFFF"
            border.color: "#D8D8DC"
            radius: 6
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                Label { text: page.titleText; color: "#1C1C1E"; font.bold: true; font.pixelSize: 16 }
                RowLayout {
                    visible: page.categoryFilter === "references"
                    Layout.fillWidth: true
                    ComboBox { id: referenceKind; model: ["actor", "location"]; Layout.preferredWidth: 110 }
                    StyledTextField { id: referenceId; placeholderText: "Reference ID"; Layout.fillWidth: true; implicitHeight: 40 }
                }
                RowLayout {
                    visible: page.categoryFilter === "references"
                    Layout.fillWidth: true
                    Button {
                        text: "Import"
                        icon.name: "document-open"
                        enabled: referenceId.text.trim().length > 0
                        Layout.fillWidth: true
                        onClicked: imageDialog.open()
                    }
                    Button {
                        text: "Rerender"
                        icon.name: "view-refresh"
                        enabled: referenceId.text.trim().length > 0
                        Layout.fillWidth: true
                        onClicked: if (vm) vm.start_reference_rerender(referenceKind.currentText, referenceId.text.trim())
                    }
                }
                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: vm ? vm.artifact_entries : []
                    delegate: ItemDelegate {
                        required property var modelData
                        visible: page.matches(modelData)
                        height: visible ? 52 : 0
                        width: ListView.view.width
                        text: modelData.path
                        icon.name: modelData.kind === "image" ? "image-x-generic" : modelData.kind === "video" ? "video-x-generic" : "text-x-generic"
                        onClicked: {
                            page.selectedPath = modelData.path
                            page.selectedKind = modelData.kind
                            if (modelData.kind === "json" && vm) vm.load_json_artifact(modelData.path)
                        }
                    }
                }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#FFFFFF"
            border.color: "#D8D8DC"
            radius: 6
            StackLayout {
                anchors.fill: parent
                anchors.margins: 12
                currentIndex: page.selectedKind === "image" ? 1 : page.selectedKind === "json" ? 2 : 0
                Label { text: page.selectedPath || "Select an artifact"; color: "#6E6E73"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; wrapMode: Text.Wrap }
                Image { source: vm ? vm.media_url(page.selectedPath) : ""; fillMode: Image.PreserveAspectFit; asynchronous: true }
                ScrollView {
                    StyledTextArea { readOnly: true; text: vm ? vm.editor_text : ""; color: "#1C1C1E"; font.family: "monospace"; wrapMode: TextEdit.NoWrap; background: null }
                }
            }
        }
    }


    FileDialog {
        id: imageDialog
        title: "Select reference image"
        nameFilters: ["Images (*.png *.jpg *.jpeg *.webp)"]
        onAccepted: {
            var extension = String(selectedFile).toLowerCase().endsWith(".webp") ? "webp"
                : String(selectedFile).toLowerCase().match(/\.jpe?g$/) ? "jpg" : "png"
            var target = "output/references/" + referenceKind.currentText + "/" + referenceId.text.trim() + "/sheet." + extension
            if (vm) vm.import_image(selectedFile, target)
        }
    }
}
