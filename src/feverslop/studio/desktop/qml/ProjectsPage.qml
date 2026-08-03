import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: page
    clip: true
    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null

    ColumnLayout {
        x: 28
        width: Math.max(0, page.availableWidth - 56)
        spacing: 18

        RowLayout {
            Layout.fillWidth: true
            Label { text: "Local project folders"; color: theme.secondaryText; font.pixelSize: 15; Layout.fillWidth: true }
            Button {
                text: "Create Project"
                icon.name: "list-add"
                icon.color: theme.cardBg
                implicitHeight: 44
                palette.buttonText: "#FFFFFF"
                background: Rectangle { color: parent.hovered ? "#666AD1" : "#5B5FC7"; radius: 6 }
                onClicked: createDialog.open()
            }
        }

        GridView {
            id: grid
            Layout.fillWidth: true
            Layout.preferredHeight: Math.max(280, Math.ceil(count / Math.max(1, Math.floor(width / 300))) * 156)
            cellWidth: Math.max(280, width / Math.max(1, Math.floor(width / 300)))
            cellHeight: 156
            model: vm ? vm.projects : []
            delegate: Item {
                required property var modelData
                width: grid.cellWidth
                height: grid.cellHeight
                Rectangle {
                    anchors.fill: parent
                    anchors.margins: 6
                    color: mouse.containsMouse ? "#3A3A3C" : theme.cardBg
                    border.color: theme.cardBorder
                    radius: 6
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 18
                        Label { text: modelData.name; color: theme.primaryText; font.bold: true; font.pixelSize: 17; elide: Text.ElideRight; Layout.fillWidth: true }
                        Label { text: modelData.id; color: theme.secondaryText; font.pixelSize: 13; elide: Text.ElideMiddle; Layout.fillWidth: true }
                        Item { Layout.fillHeight: true }
                        Label { text: String(modelData.project_type || "standard_music_video").replace(/_/g, " "); color: "#5B5FC7"; font.pixelSize: 12 }
                    }
                    MouseArea {
                        id: mouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            if (vm) vm.select_project(modelData.id)
                            root.currentPage = 1
                            root.pageTitle = "Dashboard"
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: createDialog
        modal: true
        anchors.centerIn: Overlay.overlay
        width: 520
        title: "Create Project"
        standardButtons: Dialog.Cancel
        property string songStyleError: ""
        property string movieCreateError: ""
        property bool creatingMovie: false
        onOpened: {
            projectType.currentIndex = 0
            projectName.text = ""
            projectIdea.text = ""
            songStyle.text = ""
            songStyleError = ""
            movieCreateError = ""
            duration.value = 120
            silentMode.checked = false
            pipelineMode.currentIndex = 0
        }
        ColumnLayout {
            width: parent.width
            spacing: 12
            ComboBox {
                id: projectType
                Layout.fillWidth: true
                model: ["Music video", "Full auto music video", "Movie"]
            }
            ColumnLayout {
                visible: projectType.currentIndex < 2
                Layout.fillWidth: true
                spacing: 4
                Label { text: "Video pipeline"; color: theme.primaryText; font.bold: true }
                ComboBox {
                    id: pipelineMode
                    Layout.fillWidth: true
                    textRole: "label"
                    model: [
                        { label: "Classic I2V", value: "classic" },
                        { label: "MSR (reference-guided)", value: "msr" },
                        { label: "Ingredients", value: "ingredients" }
                    ]
                }
                Label {
                    visible: projectType.currentIndex < 2
                    Layout.fillWidth: true
                    text: "Storyboard is started later as a pipeline action; it is not a video pipeline choice."
                    color: theme.secondaryText
                    font.pixelSize: 12
                    wrapMode: Text.Wrap
                }
            }
            StyledTextField {
                id: projectName
                Layout.fillWidth: true
                placeholderText: "Project name"
                placeholderTextColor: theme.secondaryText
                color: theme.primaryText
                implicitHeight: 44
                background: Rectangle { color: theme.cardBg; border.color: theme.cardBorder; radius: 4 }
            }
            StyledTextArea {
                id: projectIdea
                visible: projectType.currentIndex > 0
                Layout.fillWidth: true
                Layout.preferredHeight: 130
                placeholderText: projectType.currentIndex === 2 ? "Short story or screenplay" : "Music video idea"
                placeholderTextColor: theme.secondaryText
                color: theme.primaryText
                wrapMode: TextEdit.Wrap
                background: Rectangle { color: theme.cardBg; border.color: theme.cardBorder; radius: 4 }
            }
            ColumnLayout {
                visible: projectType.currentIndex === 1
                Layout.fillWidth: true
                spacing: 4
                Label { text: "Song style *"; color: theme.primaryText; font.bold: true }
                StyledTextField {
                    id: songStyle
                    Layout.fillWidth: true
                    placeholderText: "Song style"
                    placeholderTextColor: theme.secondaryText
                    color: theme.primaryText
                    implicitHeight: 44
                    background: Rectangle { color: theme.cardBg; border.color: theme.cardBorder; radius: 4 }
                    onTextChanged: if (text.trim().length > 0) createDialog.songStyleError = ""
                }
                Label {
                    visible: createDialog.songStyleError.length > 0
                    Layout.fillWidth: true
                    text: createDialog.songStyleError
                    color: "#C62828"
                    font.pixelSize: 12
                    wrapMode: Text.Wrap
                }
            }
            RowLayout {
                visible: projectType.currentIndex > 0
                Layout.fillWidth: true
                Label { text: "Duration"; color: theme.secondaryText }
                SpinBox { id: duration; from: 1; to: 3600; value: projectType.currentIndex === 2 ? 60 : 120; editable: true }
                Item { Layout.fillWidth: true }
                CheckBox { id: silentMode; visible: projectType.currentIndex < 2; text: "Silent mode" }
            }
            Label {
                visible: createDialog.movieCreateError.length > 0
                Layout.fillWidth: true
                text: createDialog.movieCreateError
                color: "#C62828"
                font.pixelSize: 12
                wrapMode: Text.Wrap
            }
            Button {
                text: projectType.currentIndex === 1 ? "Create and start" : "Create Project"
                icon.name: "list-add"
                icon.color: theme.cardBg
                enabled: projectName.text.trim().length > 0 && (projectType.currentIndex === 0 || projectIdea.text.trim().length > 0)
                    && !(projectType.currentIndex === 2 && vm && vm.movie_project_running)
                implicitHeight: 44
                Layout.alignment: Qt.AlignRight
                palette.buttonText: "#FFFFFF"
                background: Rectangle { color: parent.enabled ? (parent.hovered ? "#666AD1" : "#5B5FC7") : "#AEAEB2"; radius: 6 }
                onClicked: {
                    if (projectType.currentIndex === 1 && songStyle.text.trim().length === 0) {
                        createDialog.songStyleError = "Song style is required"
                        songStyle.forceActiveFocus()
                        return
                    }
                    var types = ["standard_music_video", "full_auto", "movie"]
                    var payload = {
                        project_type: types[projectType.currentIndex],
                        name: projectName.text.trim(),
                        silent_mode: silentMode.checked
                    }
                    if (projectType.currentIndex < 2) {
                        payload.pipeline_mode = pipelineMode.currentValue
                    }
                    if (projectType.currentIndex === 1) {
                        payload.idea = projectIdea.text.trim()
                        payload.song_style = songStyle.text.trim()
                        payload.duration_seconds = duration.value
                    } else if (projectType.currentIndex === 2) {
                        payload.story_text = projectIdea.text.trim()
                        payload.desired_length = duration.value
                        payload.source_type = "short_story"
                        payload.movie_mode = "scaffold"
                    }
                    if (projectType.currentIndex === 2) {
                        // Movie: async creation
                        vm.create_movie_project(payload)
                    } else {
                        // Non-movie: synchronous creation
                        var projectId = vm ? vm.create_project(payload) : ""
                        if (projectId) {
                            var currentType = projectType.currentIndex
                            if (currentType === 1) vm.start_job("full-auto", [])
                            projectType.currentIndex = 0
                            pipelineMode.currentIndex = 0
                            projectName.text = ""
                            projectIdea.text = ""
                            songStyle.text = ""
                            duration.value = 120
                            silentMode.checked = false
                            createDialog.close()
                            root.currentPage = currentType === 1 ? 2 : 1
                            root.pageTitle = currentType === 1 ? "Pipeline" : "Dashboard"
                        }
                    }
                }
            }
        }

        Connections {
            target: vm
            function onMovieProjectStarted() {
                createBusy.visible = true
                createDialog.standardButtons = Dialog.NoButton
            }
            function onMovieProjectProgress(msg) {
                createBusyText.text = msg
            }
            function onMovieProjectFinished(projectId, name) {
                createBusy.visible = false
                createDialog.standardButtons = Dialog.Cancel
                createDialog.close()
                vm.refresh_projects()
                vm.select_project(projectId)
                root.currentPage = 1
                root.pageTitle = "Dashboard"
                projectType.currentIndex = 0
                pipelineMode.currentIndex = 0
                projectName.text = ""
                projectIdea.text = ""
                songStyle.text = ""
                duration.value = 120
                silentMode.checked = false
                createDialog.creatingMovie = false
            }
            function onMovieProjectError(errorMessage) {
                createBusy.visible = false
                createDialog.standardButtons = Dialog.Cancel
                createDialog.movieCreateError = errorMessage
            }
        }

        BusyIndicator {
            id: createBusy
            visible: false
            anchors.centerIn: parent
            running: true
            z: 10
            opacity: 0.8
        }

        Label {
            id: createBusyText
            visible: createBusy.visible
            anchors.top: createBusy.bottom
            anchors.topMargin: 12
            anchors.horizontalCenter: createBusy.horizontalCenter
            text: "Creating project..."
            color: theme.secondaryText
            font.pixelSize: 12
            z: 10
        }
    }
}
