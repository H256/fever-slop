import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: page
    color: "#18181B"
    property var vm: typeof timelineViewModel !== "undefined" ? timelineViewModel : null
    property int selectedSegment: -1
    property real timelineZoom: 40
    property real timelineScroll: 0

    // Repaint canvas on local state changes
    onSelectedSegmentChanged: timelineCanvas.requestPaint()
    onTimelineZoomChanged: timelineCanvas.requestPaint()

    // Drag state for segment edge handles
    property string dragMode: ""
    property int dragSegmentIndex: -1
    property real dragStartMouseX: 0
    property real dragSegmentStart: 0
    property real dragSegmentEnd: 0

    // Time range
    readonly property real durationSegments: {
        var segs = page.vm ? page.vm.segments : []
        if (segs.length === 0) return 0
        var last = segs[segs.length - 1]
        return last.end
    }
    readonly property real totalDuration: Math.max(durationSegments, 10)

    function timeToX(t) { return 40 + t * timelineZoom }
    function xToTime(x) { return Math.max(0, (x - 40) / timelineZoom) }

    objectName: "audioTimelinePage"

    Component.onCompleted: {
        if (page.vm) page.vm.loadProject()
    }

    // Trigger canvas repaint when relevant data changes
    Connections {
        target: page.vm
        function onSegmentsChanged() { timelineCanvas.requestPaint() }
        function onBoundariesChanged() { timelineCanvas.requestPaint() }
        function onBeatsChanged() { timelineCanvas.requestPaint() }
        function onStatusChanged() { timelineCanvas.requestPaint() }
        function onUndoRedoChanged() { timelineCanvas.requestPaint() }
        function onHasChangesChanged() { timelineCanvas.requestPaint() }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 14

        // ============================================================
        // TOOLBAR: undo / redo / save / rebuild
        // ============================================================
        Rectangle {
            objectName: "timelineToolbar"
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.preferredHeight: 56

            RowLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8

                // Status pill
                Label {
                    text: page.vm ? page.vm.status : "idle"
                    color: page.vm && page.vm.error ? "#C62828" : "#A1A1AA"
                    font.pixelSize: 12
                    Layout.preferredWidth: 80
                    elide: Text.ElideRight
                }

                Label {
                    visible: page.vm && page.vm.rebuilding
                    text: "Rebuilding pipeline..."
                    color: "#93C5FF"
                    font.pixelSize: 12
                }

                Label {
                    visible: page.vm && page.vm.hasChanges
                    text: "Unsaved changes"
                    color: "#F59E0B"
                    font.pixelSize: 12
                    font.bold: true
                }
                Item { Layout.fillWidth: true }

                ToolButton {
                    objectName: "undoBtn"
                    text: "Undo"
                    icon.name: "edit-undo"
                    display: AbstractButton.TextBesideIcon
                    enabled: page.vm && page.vm.canUndo
                    implicitHeight: 36
                    ToolTip.visible: hovered
                    ToolTip.text: "Undo last edit"
                    onClicked: if (page.vm) page.vm.undo()
                }

                ToolButton {
                    objectName: "redoBtn"
                    text: "Redo"
                    icon.name: "edit-redo"
                    display: AbstractButton.TextBesideIcon
                    enabled: page.vm && page.vm.canRedo
                    implicitHeight: 36
                    ToolTip.visible: hovered
                    ToolTip.text: "Redo last edit"
                    onClicked: if (page.vm) page.vm.redo()
                }

                Button {
                    objectName: "saveBtn"
                    text: "Save"
                    icon.name: "document-save"
                    enabled: page.vm && page.vm.hasChanges
                    implicitHeight: 36
                    onClicked: { if (page.vm) page.vm.save(); page.selectedSegment = -1 }
                }

                Button {
                    objectName: "rebuildBtn"
                    text: "Rebuild Pipeline"
                    icon.name: "view-refresh"
                    enabled: page.vm && !page.vm.rebuilding
                    implicitHeight: 36
                    palette.button: "#4F46E5"
                    palette.buttonText: "#FFFFFF"
                    onClicked: if (page.vm) page.vm.rebuildPipeline()
                }
            }
        }

        // ============================================================
        // ZOOM & TIME RULER CONTROLS
        // ============================================================
        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Label {
                text: "Zoom"
                color: "#A1A1AA"
                font.pixelSize: 12
                verticalAlignment: Text.AlignVCenter
            }
            Slider {
                from: 12
                to: 120
                value: page.timelineZoom
                onMoved: page.timelineZoom = value
                Layout.fillWidth: true
                Layout.maximumWidth: 240
                background: Rectangle {
                    color: "#3F3F46"
                    height: 6
                    radius: 3
                }
            }
            Label {
                text: page.timelineZoom.toFixed(0) + " px/s"
                color: "#D4D4D8"
                font.pixelSize: 12
                verticalAlignment: Text.AlignVCenter
            }
            Item { Layout.fillWidth: true }
            Label {
                text: page.totalDuration.toFixed(2) + " s total"
                color: "#8E8E93"
                font.pixelSize: 12
                verticalAlignment: Text.AlignVCenter
            }
        }

        // ============================================================
        // TIMELINE CANVAS (segments, boundaries, beats)
        // ============================================================
        ScrollView {
            id: timelineScroll
            Layout.fillWidth: true
            Layout.preferredHeight: 180
            ScrollBar.vertical.policy: ScrollBar.AlwaysOff
            ScrollBar.horizontal: ScrollBar {
                contentItem: Rectangle {
                    implicitWidth: 8
                    radius: 4
                    color: "#52525B"
                }
                background: Rectangle { color: "#1C1C1E" }
            }

            Rectangle {
                id: timelineCanvasRoot
                color: "#1C1C1E"
                width: Math.ceil(page.totalDuration * page.timelineZoom) + 200
                height: timelineScroll.contentHeight

                Rectangle {
                    id: clipper
                    anchors.fill: parent
                    clip: true

                    Rectangle {
                        id: timelineCanvasArea
                        anchors.fill: parent
                        color: "#212126"

                        Canvas {
                            id: timelineCanvas
                            anchors.fill: parent
                            renderStrategy: Canvas.Immediate

                            function paint() {
                                var ctx = timelineCanvas.getContext("2d")
                                ctx.reset()
                                var w = timelineCanvas.width
                                var h = timelineCanvas.height

                                // Background grid (time ticks)
                                var gridEvery = 1
                                if (page.timelineZoom < 20) gridEvery = 5
                                if (page.timelineZoom >= 60) gridEvery = 0.5

                                for (var t = gridEvery; t <= page.totalDuration; t += gridEvery) {
                                    var gx = page.timeToX(t)
                                    ctx.strokeStyle = "#2B2B30"
                                    ctx.lineWidth = 1
                                    ctx.beginPath()
                                    ctx.moveTo(gx, 0)
                                    ctx.lineTo(gx, h)
                                    ctx.stroke()
                                }

                                // Time labels along the left edge
                                ctx.fillStyle = "#6E6E73"
                                ctx.font = "11px monospace"
                                ctx.textBaseline = "middle"
                                for (var tl = 0; tl <= page.totalDuration; tl += 1) {
                                    var tlx = page.timeToX(tl)
                                    ctx.fillText(tl.toFixed(1) + "s", 2, tlx)
                                }

                                // ---- SEGMENTS ----
                                var segBarY = 28
                                var segBarH = 64
                                var segs = page.vm ? page.vm.segments : []
                                for (var si = 0; si < segs.length; ++si) {
                                    var s = segs[si]
                                    var sx = page.timeToX(s.start)
                                    var sw = Math.max(4, (s.end - s.start) * page.timelineZoom)
                                    var isSel = si === page.selectedSegment

                                    // Color by kind
                                    ctx.fillStyle = s.kind === "chorus" ? "#3366CC"
                                        : s.kind === "bridge" ? "#CC8833"
                                        : s.kind === "outro" ? "#9955BB"
                                        : "#44AA66"
                                    ctx.globalAlpha = isSel ? 1 : 0.82
                                    ctx.strokeStyle = isSel ? "#FFFFFF" : "#2A2A30"
                                    ctx.lineWidth = isSel ? 2 : 1
                                    roundRect(ctx, sx, segBarY, sw, segBarH, 4)
                                    ctx.globalAlpha = 1

                                    // Draft overlay pattern
                                    if (s.is_draft) {
                                        ctx.save()
                                        roundRect(ctx, sx, segBarY, sw, segBarH, 4)
                                        ctx.clip()
                                        ctx.strokeStyle = "rgba(255,255,255,0.18)"
                                        ctx.lineWidth = 1
                                        for (var d = sx - sw; d < sx + sw + 10; d += 8) {
                                            ctx.beginPath()
                                            ctx.moveTo(d, segBarY)
                                            ctx.lineTo(d + 12, segBarY + segBarH)
                                            ctx.stroke()
                                        }
                                        ctx.restore()
                                    }

                                    // Segment label
                                    ctx.fillStyle = "#FFFFFF"
                                    ctx.font = "bold 11px sans-serif"
                                    ctx.textBaseline = "top"
                                    var label = s.kind.charAt(0).toUpperCase() + s.kind.slice(1)
                                    if (s.text && s.text.length > 0) label = s.text
                                    ctx.fillText(label, sx + 5, segBarY + 5)

                                    // Lyrics line (smaller, below label)
                                    if (s.lyrics_line && s.lyrics_line.length > 0) {
                                        ctx.fillStyle = "#D0D0D8"
                                        ctx.font = "10px sans-serif"
                                        ctx.fillText(s.lyrics_line, sx + 5, segBarY + 22)
                                    }

                                    // Time label below
                                    ctx.fillStyle = "#8E8E93"
                                    ctx.font = "10px monospace"
                                    ctx.textBaseline = "bottom"
                                    ctx.fillText(
                                        s.start.toFixed(1) + "-" + s.end.toFixed(1),
                                        sx + 5, segBarY + segBarH - 4
                                    )

                                    // Drag handles (left/right edges)
                                    var handleW = 6
                                    ctx.fillStyle = isSel ? "#FFFFFF" : "#FFFFFFCC"
                                    // Left handle
                                    roundRect(ctx, sx - 2, segBarY + segBarH / 2 - 8, handleW, 16, 3)
                                    // Right handle
                                    roundRect(ctx, sx + sw - handleW + 2, segBarY + segBarH / 2 - 8, handleW, 16, 3)
                                }

                                // ---- SCENE BOUNDARIES ----
                                var bndY = segBarY + segBarH + 10
                                var bndH = 14
                                var bnds = page.vm ? page.vm.boundaries : []
                                for (var bi = 0; bi < bnds.length; ++bi) {
                                    var b = bnds[bi]
                                    var bx = page.timeToX(b.start)
                                    var bw = Math.max(4, (b.end - b.start) * page.timelineZoom)

                                    ctx.fillStyle = "#3A4A3A"
                                    ctx.globalAlpha = 0.5
                                    roundRect(ctx, bx, bndY, bw, bndH, 2)
                                    ctx.globalAlpha = 1

                                    ctx.fillStyle = "#7A9A7A"
                                    ctx.font = "9px monospace"
                                    ctx.textBaseline = "top"
                                    ctx.fillText("scene", bx + 2, bndY + 2)
                                }

                                // Boundary vertical lines
                                ctx.strokeStyle = "#5A8A5A"
                                ctx.lineWidth = 1
                                ctx.setLineDash([4, 3])
                                for (var bl = 0; bl < bnds.length; ++bl) {
                                    var bd = bnds[bl]
                                    var blx = page.timeToX(bd.start)
                                    ctx.beginPath()
                                    ctx.moveTo(blx, 0)
                                    ctx.lineTo(blx, segBarY + segBarH)
                                    ctx.stroke()
                                }
                                ctx.setLineDash([])

                                // ---- BEAT TRACK ----
                                var beatY = bndY + bndH + 10
                                var beats = page.vm ? page.vm.beats : []
                                for (var br = 0; br < beats.length; ++br) {
                                    var be = beats[br]
                                    var beX = page.timeToX(be.time_s)
                                    var size = 3 + be.confidence * 5

                                    // Triangle
                                    ctx.fillStyle = "#E8A030"
                                    ctx.globalAlpha = 0.5 + be.confidence * 0.5
                                    ctx.beginPath()
                                    ctx.moveTo(beX, beatY)
                                    ctx.lineTo(beX - size, beatY + size * 1.5)
                                    ctx.lineTo(beX + size, beatY + size * 1.5)
                                    ctx.closePath()
                                    ctx.fill()
                                    ctx.globalAlpha = 1

                                    // Label for high-confidence beats
                                    if (be.confidence > 0.7 && be.label) {
                                        ctx.fillStyle = "#D09020"
                                        ctx.font = "8px monospace"
                                        ctx.textBaseline = "top"
                                        ctx.fillText(be.label, beX - 8, beatY + size * 1.5 + 2)
                                    }
                                }
                            }

                            function roundRect(ctx, x, y, w, h, r) {
                                ctx.beginPath()
                                ctx.moveTo(x + r, y)
                                ctx.lineTo(x + w - r, y)
                                ctx.quadraticCurveTo(x + w, y, x + w, y + r)
                                ctx.lineTo(x + w, y + h - r)
                                ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
                                ctx.lineTo(x + r, y + h)
                                ctx.quadraticCurveTo(x, y + h, x, y + h - r)
                                ctx.lineTo(x, y + r)
                                ctx.quadraticCurveTo(x, y, x + r, y)
                                ctx.closePath()
                                ctx.fill()
                                if (ctx.lineWidth > 0) ctx.stroke()
                            }
                        }

                        // Invisible interactive layer for mouse interaction
                        MouseArea {
                            id: timelineInteraction
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.ArrowCursor

                            function hitSegment(mx) {
                                var time = page.xToTime(mx)
                                var segs = page.vm ? page.vm.segments : []
                                var handleWidth = 6 / page.timelineZoom // time units

                                for (var i = 0; i < segs.length; ++i) {
                                    var s = segs[i]
                                    // Check right handle (within 6px of end)
                                    if (Math.abs(time - s.end) <= handleWidth)
                                        return { index: i, action: "resize-end" }
                                    // Check left handle (within 6px of start)
                                    if (Math.abs(time - s.start) <= handleWidth)
                                        return { index: i, action: "resize-start" }
                                    // Check segment body
                                    if (time >= s.start && time <= s.end)
                                        return { index: i, action: "select" }
                                }
                                return null
                            }

                            onPressed: function(mouse) {
                                var hit = timelineInteraction.hitSegment(mouse.x)
                                if (hit) {
                                    if (hit.action === "resize-end") {
                                        page.dragMode = "resize-end"
                                        page.dragSegmentIndex = hit.index
                                        page.dragStartMouseX = mouse.x
                                        var seg = page.vm.segments[hit.index]
                                        page.dragSegmentEnd = seg.end
                                        page.dragSegmentStart = seg.start
                                        mouse.accepted = true
                                        timelineInteraction.cursorShape = Qt.SizeHorCursor
                                    } else if (hit.action === "resize-start") {
                                        page.dragMode = "resize-start"
                                        page.dragSegmentIndex = hit.index
                                        page.dragStartMouseX = mouse.x
                                        var seg2 = page.vm.segments[hit.index]
                                        page.dragSegmentEnd = seg2.end
                                        page.dragSegmentStart = seg2.start
                                        mouse.accepted = true
                                        timelineInteraction.cursorShape = Qt.SizeHorCursor
                                    } else {
                                        page.selectedSegment = hit.index
                                        page.dragMode = ""
                                        mouse.accepted = true
                                    }
                                } else {
                                    page.selectedSegment = -1
                                    page.dragMode = ""
                                }
                                timelineCanvas.requestPaint()
                            }

                            onPositionChanged: function(mouse) {
                                if (!page.dragMode) {
                                    // Update cursor when not dragging
                                    var hit = timelineInteraction.hitSegment(mouse.x)
                                    if (hit) {
                                        if (hit.action.indexOf("resize") !== -1)
                                            timelineInteraction.cursorShape = Qt.SizeHorCursor
                                        else
                                            timelineInteraction.cursorShape = Qt.PointingHandCursor
                                    } else {
                                        timelineInteraction.cursorShape = Qt.ArrowCursor
                                    }
                                    return
                                }

                                var dx = mouse.x - page.dragStartMouseX
                                var dt = dx / page.timelineZoom

                                if (page.dragMode === "resize-end") {
                                    page.dragSegmentEnd = Math.max(
                                        page.dragSegmentStart + 0.1,
                                        page.dragSegmentEnd + dt
                                    )
                                } else if (page.dragMode === "resize-start") {
                                    page.dragSegmentStart = Math.min(
                                        page.dragSegmentEnd - 0.1,
                                        Math.max(0, page.dragSegmentStart + dt)
                                    )
                                }
                                timelineCanvas.requestPaint()
                            }

                            onReleased: function(mouse) {
                                if (!page.dragMode || page.dragSegmentIndex < 0) {
                                    page.dragMode = ""
                                    timelineInteraction.cursorShape = Qt.ArrowCursor
                                    return
                                }

                                var idx = page.dragSegmentIndex
                                var startDelta = page.dragSegmentStart - page.vm.segments[idx].start
                                var endDelta = page.dragSegmentEnd - page.vm.segments[idx].end

                                if (Math.abs(startDelta) > 0.01 || Math.abs(endDelta) > 0.01) {
                                    page.vm.editSegment(idx, startDelta, endDelta, "", "")
                                }

                                page.dragMode = ""
                                page.dragSegmentIndex = -1
                                timelineInteraction.cursorShape = Qt.ArrowCursor
                            }
                        }
                    }
                }
            }
        }

        // ============================================================
        // ACTION BAR: split / merge / add beat / add boundary
        // ============================================================
        Rectangle {
            objectName: "timelineActionBar"
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.preferredHeight: 50

            RowLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8

                Label {
                    text: "Actions"
                    color: "#8E8E93"
                    font.bold: true
                    font.pixelSize: 11
                    verticalAlignment: Text.AlignVCenter
                }
                Item { Layout.fillWidth: true }

                Button {
                    objectName: "splitSegmentBtn"
                    text: "Split at cursor"
                    enabled: page.selectedSegment >= 0
                    implicitHeight: 32
                    onClicked: {
                        if (page.selectedSegment < 0) return
                        var seg = page.vm.segments[page.selectedSegment]
                        var mid = (seg.start + seg.end) / 2
                        page.vm.splitSegment(page.selectedSegment, mid)
                        page.selectedSegment = -1
                    }
                }

                Button {
                    objectName: "mergeSegmentsBtn"
                    text: "Merge " + (page.selectedSegment + 1)
                    enabled: page.selectedSegment >= 0
                    implicitHeight: 32
                    onClicked: {
                        if (page.selectedSegment < 0) return
                        page.vm.mergeSegments(page.selectedSegment, 2)
                        page.selectedSegment = Math.max(0, page.selectedSegment - 1)
                    }
                }

                Button {
                    objectName: "addBoundaryBtn"
                    text: "Add scene boundary"
                    implicitHeight: 32
                    onClicked: dialogAddBoundary.visible = true
                }

                Button {
                    objectName: "addBeatBtn"
                    text: "Add beat"
                    implicitHeight: 32
                    onClicked: dialogAddBeat.visible = true
                }
            }
        }

        // ============================================================
        // SEGMENT LIST: scrollable list of all segments
        // ============================================================
        Rectangle {
            objectName: "segmentListContainer"
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.preferredHeight: 100

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 6

                Label {
                    text: "Segments (" + (page.vm ? page.vm.segments.length : 0) + ")"
                    color: "#A1A1AA"
                    font.bold: true
                    font.pixelSize: 11
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: page.vm ? page.vm.segments : null
                    spacing: 3
                    boundsBehavior: Flickable.StopAtBounds

                    delegate: Rectangle {
                        required property int index
                        required property real start
                        required property real end
                        required property string kind
                        required property string text
                        required property string lyrics_line
                        required property bool is_draft

                        readonly property bool isSelected: page.selectedSegment === index

                        width: parent ? parent.width : 0
                        height: 30
                        color: isSelected ? "#3A3A50" : "#202024"
                        border.color: isSelected ? "#5B6EE0" : "#2A2A30"
                        radius: 4

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 4
                            spacing: 8

                            Rectangle {
                                width: 4
                                Layout.fillHeight: true
                                radius: 2
                                color: model.kind === "chorus" ? "#3366CC"
                                    : model.kind === "bridge" ? "#CC8833"
                                    : model.kind === "outro" ? "#9955BB"
                                    : "#44AA66"
                            }
                            Label {
                                text: model.start.toFixed(2) + "-" + model.end.toFixed(2)
                                color: "#D4D4D8"
                                font.pixelSize: 11
                                font.family: "monospace"
                                Layout.preferredWidth: 100
                            }
                            Label {
                                text: model.kind
                                color: "#A1A1AA"
                                font.pixelSize: 11
                                font.bold: true
                                Layout.preferredWidth: 60
                            }
                            Label {
                                text: model.text || ""
                                color: "#C7C7CC"
                                font.pixelSize: 11
                                elide: Text.ElideMiddle
                                Layout.fillWidth: true
                            }
                            Label {
                                visible: model.is_draft
                                text: "DRAFT"
                                color: "#F59E0B"
                                font.pixelSize: 9
                                font.bold: true
                            }
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: page.selectedSegment = index
                        }
                    }
                }
            }
        }

        // ============================================================
        // DETAILS PANEL: edit selected segment properties
        // ============================================================
        Rectangle {
            id: panel
            objectName: "segmentDetailsPanel"
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.fillHeight: true

            readonly property var selData: page.selectedSegment >= 0
                ? (page.vm ? page.vm.segments[page.selectedSegment] : null)
                : null
            readonly property real selectedStart: selData ? selData.start : 0
            readonly property real selectedEnd: selData ? selData.end : 0
            readonly property string selectedKind: selData ? selData.kind : ""
            readonly property bool selectedDraft: selData ? selData.is_draft : false
            readonly property string selectedLyrics: selData ? selData.lyrics_line : ""
            readonly property string selectedNotes: selData ? selData.notes : ""

            ColumnLayout {
                id: detailLayout
                anchors.fill: parent
                anchors.margins: 14
                spacing: 10

                Label {
                    text: page.selectedSegment >= 0
                        ? "Selected segment details"
                        : "No segment selected"
                    color: page.selectedSegment >= 0 ? "#F4F4F5" : "#6E6E73"
                    font.bold: true
                    font.pixelSize: 13
                }

                // Detail content — only shown when a segment is selected
                ColumnLayout {
                    Layout.fillWidth: true
                    visible: !!panel.selData
                    spacing: 10

                    GridLayout {
                        columns: 2
                        columnSpacing: 14
                        rowSpacing: 8

                        Label { text: "Start (s)"; color: "#8E8E93"; font.pixelSize: 11 }
                        Label {
                            text: panel.selectedStart.toFixed(3)
                            color: "#D4D4D8"
                            font.pixelSize: 11
                            font.family: "monospace"
                        }

                        Label { text: "End (s)"; color: "#8E8E93"; font.pixelSize: 11 }
                        Label {
                            text: panel.selectedEnd.toFixed(3)
                            color: "#D4D4D8"
                            font.pixelSize: 11
                            font.family: "monospace"
                        }

                        Label { text: "Kind"; color: "#8E8E93"; font.pixelSize: 11 }
                        Label {
                            text: panel.selectedKind
                            color: "#D4D4D8"
                            font.pixelSize: 11
                            font.bold: true
                        }

                        Label { text: "Draft"; color: "#8E8E93"; font.pixelSize: 11 }
                        Label {
                            text: panel.selectedDraft ? "Yes" : "No"
                            color: panel.selectedDraft ? "#F59E0B" : "#D4D4D8"
                            font.pixelSize: 11
                        }
                    }

                    Item { height: 6 }

                    Label { text: "Lyrics line"; color: "#8E8E93"; font.pixelSize: 11 }
                    StyledTextField {
                        id: fieldLyrics
                        placeholderText: "Enter lyrics or leave empty"
                        placeholderTextColor: "#6E6E73"
                        text: panel.selectedLyrics
                        color: "#F4F4F5"
                        selectionColor: "#3B3F8C"
                        selectedTextColor: "#FFFFFF"
                        Layout.fillWidth: true
                        background: Rectangle {
                            color: "#18181B"
                            border.color: fieldLyrics.activeFocus ? "#818CF8" : "#52525B"
                            radius: 6
                        }
                    }

                    Label { text: "Notes"; color: "#8E8E93"; font.pixelSize: 11 }
                    StyledTextField {
                        id: fieldNotes
                        placeholderText: "Editor notes for this segment"
                        placeholderTextColor: "#6E6E73"
                        text: panel.selectedNotes
                        color: "#F4F4F5"
                        selectionColor: "#3B3F8C"
                        selectedTextColor: "#FFFFFF"
                        Layout.fillWidth: true
                        background: Rectangle {
                            color: "#18181B"
                            border.color: fieldNotes.activeFocus ? "#818CF8" : "#52525B"
                            radius: 6
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Button {
                            objectName: "applySegmentEditBtn"
                            text: "Apply text edits"
                            enabled: true
                            implicitHeight: 34
                            onClicked: {
                                if (page.selectedSegment < 0 || !page.vm) return
                                page.vm.editSegment(
                                    page.selectedSegment,
                                    0,
                                    0,
                                    fieldLyrics.text,
                                    fieldNotes.text
                                )
                            }
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: "Drag segment edges on the timeline to adjust timing"
                            color: "#6E6E73"
                            font.pixelSize: 10
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                            Layout.maximumWidth: 260
                        }
                    }
                }
            }
        }

        // ============================================================
        // BEAT TABLE: list of beat markers
        // ============================================================
        Rectangle {
            objectName: "beatTableContainer"
            color: "#27272A"
            border.color: "#3F3F46"
            radius: 10
            Layout.fillWidth: true
            Layout.preferredHeight: 70

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 4

                Label {
                    text: "Beat markers (" + (page.vm ? page.vm.beats.length : 0) + ")"
                    color: "#A1A1AA"
                    font.bold: true
                    font.pixelSize: 11
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: page.vm ? page.vm.beats : null
                    spacing: 2
                    boundsBehavior: Flickable.StopAtBounds

                    delegate: Rectangle {
                        required property real time_s
                        required property string label
                        required property real confidence

                        width: parent ? parent.width : 0
                        height: 24
                        color: "#202024"
                        radius: 3
                        opacity: 0.6 + confidence * 0.4

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 6
                            spacing: 8

                            Label {
                                text: time_s.toFixed(2) + "s"
                                color: "#D4D4D8"
                                font.pixelSize: 10
                                font.family: "monospace"
                                Layout.preferredWidth: 60
                            }
                            Label {
                                text: label || ""
                                color: "#D09020"
                                font.pixelSize: 10
                                Layout.fillWidth: true
                                elide: Text.ElideMiddle
                            }
                            Label {
                                text: "conf " + confidence.toFixed(1)
                                color: "#8E8E93"
                                font.pixelSize: 10
                                Layout.preferredWidth: 70
                            }
                        }
                    }
                }
            }
        }
    }

    // ============================================================
    // DIALOG: Add beat
    // ============================================================
    Rectangle {
        id: dialogAddBeat
        visible: false
        anchors.centerIn: parent
        width: 320
        height: 180
        color: "#27272A"
        border.color: "#3F3F46"
        radius: 10
        z: 100

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 18
            spacing: 12

            Label { text: "Add Beat Marker"; color: "#F4F4F5"; font.bold: true }

            Label { text: "Time (seconds)"; color: "#8E8E93"; font.pixelSize: 11 }
            StyledTextField {
                id: beatTimeField
                placeholderText: "0.00"
                placeholderTextColor: "#6E6E73"
                color: "#F4F4F5"
                validator: DoubleValidator { bottom: 0 }
                background: Rectangle {
                    color: "#18181B"
                    border.color: beatTimeField.activeFocus ? "#818CF8" : "#52525B"
                    radius: 6
                }
            }

            Label { text: "Label (optional)"; color: "#8E8E93"; font.pixelSize: 11 }
            StyledTextField {
                id: beatLabelField
                placeholderText: "e.g. drop, fill, snare"
                placeholderTextColor: "#6E6E73"
                color: "#F4F4F5"
                background: Rectangle {
                    color: "#18181B"
                    border.color: beatLabelField.activeFocus ? "#818CF8" : "#52525B"
                    radius: 6
                }
            }

            Label { text: "Confidence (0-1)"; color: "#8E8E93"; font.pixelSize: 11 }
            Slider {
                id: beatConfSlider
                from: 0
                to: 1
                value: 0.8
                stepSize: 0.1
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                Item { Layout.fillWidth: true }
                Button {
                    text: "Add"
                    implicitHeight: 32
                    onClicked: {
                        if (page.vm) {
                            page.vm.addBeat(
                                Number(beatTimeField.text) || 0,
                                beatLabelField.text,
                                beatConfSlider.value
                            )
                        }
                        dialogAddBeat.visible = false
                        beatTimeField.text = ""
                        beatLabelField.text = ""
                    }
                }
                Button {
                    text: "Cancel"
                    implicitHeight: 32
                    onClicked: dialogAddBeat.visible = false
                }
            }
        }

        MouseArea {
            anchors.fill: parent
            anchors.margins: -20
            onClicked: dialogAddBeat.visible = false
            propagateComposedEvents: true
        }
    }

    // ============================================================
    // DIALOG: Add scene boundary
    // ============================================================
    Rectangle {
        id: dialogAddBoundary
        visible: false
        anchors.centerIn: parent
        width: 360
        height: 210
        color: "#27272A"
        border.color: "#3F3F46"
        radius: 10
        z: 100

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 18
            spacing: 12

            Label { text: "Add Scene Boundary"; color: "#F4F4F5"; font.bold: true }

            Label { text: "Start (seconds)"; color: "#8E8E93"; font.pixelSize: 11 }
            StyledTextField {
                id: bndStartField
                placeholderText: "0.00"
                placeholderTextColor: "#6E6E73"
                color: "#F4F4F5"
                validator: DoubleValidator { bottom: 0 }
                background: Rectangle {
                    color: "#18181B"
                    border.color: bndStartField.activeFocus ? "#818CF8" : "#52525B"
                    radius: 6
                }
            }

            Label { text: "End (seconds)"; color: "#8E8E93"; font.pixelSize: 11 }
            StyledTextField {
                id: bndEndField
                placeholderText: "0.00"
                placeholderTextColor: "#6E6E73"
                color: "#F4F4F5"
                validator: DoubleValidator { bottom: 0 }
                background: Rectangle {
                    color: "#18181B"
                    border.color: bndEndField.activeFocus ? "#818CF8" : "#52525B"
                    radius: 6
                }
            }

            Label { text: "Reason"; color: "#8E8E93"; font.pixelSize: 11 }
            StyledTextField {
                id: bndReasonField
                placeholderText: "e.g. chorus change, bridge transition"
                placeholderTextColor: "#6E6E73"
                color: "#F4F4F5"
                background: Rectangle {
                    color: "#18181B"
                    border.color: bndReasonField.activeFocus ? "#818CF8" : "#52525B"
                    radius: 6
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                Item { Layout.fillWidth: true }
                Button {
                    text: "Add"
                    implicitHeight: 32
                    onClicked: {
                        if (page.vm) {
                            page.vm.addSceneBoundary(
                                Number(bndStartField.text) || 0,
                                Number(bndEndField.text) || 0,
                                bndReasonField.text
                            )
                        }
                        dialogAddBoundary.visible = false
                        bndStartField.text = ""
                        bndEndField.text = ""
                        bndReasonField.text = ""
                    }
                }
                Button {
                    text: "Cancel"
                    implicitHeight: 32
                    onClicked: dialogAddBoundary.visible = false
                }
            }
        }

        MouseArea {
            anchors.fill: parent
            anchors.margins: -20
            onClicked: dialogAddBoundary.visible = false
            propagateComposedEvents: true
        }
    }
}
