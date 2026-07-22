import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root
    width: 1440
    height: 900
    minimumWidth: 1000
    minimumHeight: 680
    visible: true
    title: "FeverSlop Studio"
    color: "#1C1C1E"

    readonly property var vm: typeof studioViewModel !== "undefined" ? studioViewModel : null
    property int currentPage: 0
    property string pageTitle: "Projects"

    Connections {
        target: root.vm
        function onCurrentProjectChanged() {
            if (root.currentPage === 11
                    && root.vm.current_project.project_type === "movie") {
                root.currentPage = 1
                root.pageTitle = "Dashboard"
            }
        }
    }

    component NavButton: Button {
        required property int page
        readonly property string fallbackIcon: {
            if (page === 0 || page === 6) return "\u25a0"
            if (page === 1) return "\u25a6"
            if (page === 2 || page === 9) return "\u25b6"
            if (page === 3 || page === 7) return "\u2261"
            if (page === 4 || page === 8) return "\u25a3"
            return "\u25cf"
        }
        flat: true
        Layout.fillWidth: true
        Layout.preferredHeight: 44
        leftPadding: 14
        rightPadding: 12
        display: AbstractButton.TextBesideIcon
        font.pixelSize: 14
        palette.buttonText: checked ? "#FFFFFF" : "#C7C7CC"
        palette.disabled.buttonText: "#8E8E93"
        icon.color: enabled ? (checked ? "#FFFFFF" : "#C7C7CC") : "#8E8E93"
        contentItem: RowLayout {
            spacing: 10
            Label {
                text: parent.parent.fallbackIcon
                color: parent.parent.enabled
                    ? (parent.parent.checked ? "#FFFFFF" : "#C7C7CC")
                    : "#8E8E93"
                font.pixelSize: 18
                horizontalAlignment: Text.AlignHCenter
                Layout.preferredWidth: 24
            }
            Label {
                text: parent.parent.text
                color: parent.parent.enabled
                    ? (parent.parent.checked ? "#FFFFFF" : "#C7C7CC")
                    : "#8E8E93"
                font: parent.parent.font
                elide: Text.ElideRight
                verticalAlignment: Text.AlignVCenter
                Layout.fillWidth: true
            }
        }
        background: Rectangle {
            color: checked ? "#3A3A40" : parent.hovered ? "#303034" : "transparent"
            radius: 6
            Rectangle {
                visible: parent.parent.checked
                width: 3
                height: 24
                radius: 2
                color: "#7B83EB"
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
            }
        }
        onClicked: {
            root.currentPage = page
            root.pageTitle = text
        }
    }

    component ChromeLabel: Label {
        color: "#E5E5EA"
        font.pixelSize: 13
    }

    component WorkspaceButton: Button {
        implicitHeight: 44
        leftPadding: 16
        rightPadding: 16
        palette.buttonText: "#FFFFFF"
        background: Rectangle {
            color: parent.down ? "#4A4EB3" : parent.hovered ? "#666AD1" : "#5B5FC7"
            radius: 6
        }
    }

    header: ToolBar {
        height: 56
        background: Rectangle { color: "#252528" }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 16
            spacing: 12
            Label {
                text: "FS"
                color: "white"
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                background: Rectangle { color: "#5B5FC7"; radius: 5 }
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
            }
            Label { text: "FeverSlop Studio"; color: "#FFFFFF"; font.bold: true; font.pixelSize: 16 }
            Rectangle { width: 1; Layout.fillHeight: true; color: "#3A3A3C" }
            ChromeLabel {
                text: vm && vm.current_project_id ? vm.current_project.name : "Project workspace"
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            ToolButton {
                icon.name: "view-refresh"
                icon.color: "#E5E5EA"
                display: AbstractButton.IconOnly
                ToolTip.visible: hovered
                ToolTip.text: "Refresh projects"
                onClicked: if (vm) vm.refresh_projects()
            }
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            objectName: "projectSidebar"
            color: "#252528"
            Layout.preferredWidth: 240
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 4

                ChromeLabel { text: "WORKSPACE"; color: "#8E8E93"; font.bold: true; font.pixelSize: 11; leftPadding: 10 }
                NavButton { text: "Projects"; icon.name: "folder"; page: 0; checked: root.currentPage === page }
                NavButton { text: "Studio Settings"; icon.name: "settings-configure"; page: 10; checked: root.currentPage === page }

                Rectangle { Layout.fillWidth: true; height: 1; color: "#3A3A3C"; Layout.topMargin: 6; Layout.bottomMargin: 6 }
                ChromeLabel {
                    text: vm && vm.current_project_id ? vm.current_project_id.toUpperCase() : "NO PROJECT SELECTED"
                    color: "#8E8E93"
                    font.bold: true
                    font.pixelSize: 11
                    leftPadding: 10
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
                NavButton { text: "Dashboard"; icon.name: "view-dashboard"; page: 1; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton { text: "Pipeline"; icon.name: "media-playback-start"; page: 2; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton {
                    objectName: "sceneWorkspaceNavigation"
                    text: "Scene Workspace"
                    icon.name: "view-grid"
                    page: 11
                    checked: root.currentPage === page
                    enabled: vm && vm.current_project_id
                        && vm.current_project.project_type !== "movie"
                }
                NavButton { text: "Render Plan"; icon.name: "view-list-details"; page: 3; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton { text: "References"; icon.name: "image-x-generic"; page: 4; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton { text: "Project Settings"; icon.name: "document-properties"; page: 5; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton { text: "Artifacts"; icon.name: "folder-documents"; page: 6; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton { text: "Queue"; icon.name: "view-list-tree"; page: 7; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton { text: "Review"; icon.name: "video-x-generic"; page: 8; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                NavButton { text: "Final Video"; icon.name: "media-playback-start"; page: 9; checked: root.currentPage === page; enabled: vm && vm.current_project_id }
                Item { Layout.fillHeight: true }
                ChromeLabel { text: "Native Qt Studio"; color: "#6E6E73"; leftPadding: 10 }
            }
        }

        ColumnLayout {
            objectName: "workspace"
            spacing: 0
            Layout.fillWidth: true
            Layout.fillHeight: true

            Rectangle {
                color: "#F5F5F7"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 0

                    Rectangle {
                        objectName: "workspaceHeader"
                        color: root.currentPage === 11 ? "#202024" : "#FFFFFF"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 72
                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 28
                            anchors.rightMargin: 28
                            Label {
                                text: root.pageTitle
                                color: root.currentPage === 11 ? "#F4F4F5" : "#1C1C1E"
                                font.pixelSize: 24
                                font.bold: true
                                Layout.fillWidth: true
                            }
                            Label { visible: vm && vm.error; text: vm ? vm.error : ""; color: "#C62828"; font.pixelSize: 13; elide: Text.ElideRight; Layout.maximumWidth: 520 }
                        }
                        Rectangle {
                            anchors.bottom: parent.bottom
                            width: parent.width
                            height: 1
                            color: root.currentPage === 11 ? "#3F3F46" : "#D8D8DC"
                        }
                    }

                    StackLayout {
                        currentIndex: root.currentPage
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        ProjectsPage {}
                        DashboardPage {}
                        PipelinePage {}
                        JsonWorkspace { heading: "Render Plan"; defaultPath: "render_plan.json" }
                        ArtifactPage { categoryFilter: "references"; titleText: "Reference assets" }
                        JsonWorkspace { heading: "Project Configuration"; defaultPath: "config.json" }
                        ArtifactPage { categoryFilter: ""; titleText: "Project artifacts" }
                        QueuePage {}
                        ReviewPage {}
                        MediaPage { reviewMode: false }
                        PlaceholderPage { heading: "Studio Settings"; detail: "The projects root is selected at application startup." }
                        SceneWorkspacePage {}
                    }
                }
            }

            Rectangle {
                objectName: "jobPanel"
                color: "#1C1C1E"
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 16
                    ColumnLayout {
                        Layout.preferredWidth: 210
                        Label { text: "JOB ACTIVITY"; color: "#8E8E93"; font.bold: true; font.pixelSize: 11 }
                        ChromeLabel { text: vm && vm.active_job.id ? vm.active_job.action : "No active job"; font.pixelSize: 15 }
                        ChromeLabel {
                            text: vm && vm.active_job.id
                                ? (vm.active_job.status + "  " + (vm.active_job.overall_progress || 0) + "%")
                                : "Pipeline output appears here"
                            color: "#8E8E93"
                        }
                    }
                    Rectangle { width: 1; Layout.fillHeight: true; color: "#3A3A3C" }
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        TextArea {
                            readOnly: true
                            text: vm ? vm.job_logs : "Ready."
                            color: "#C7C7CC"
                            font.family: "monospace"
                            font.pixelSize: 12
                            background: null
                        }
                    }
                }
            }
        }
    }
}
