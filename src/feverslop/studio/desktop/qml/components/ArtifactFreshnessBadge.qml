import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/**
 * ArtifactFreshnessBadge - displays current/stale/unknown freshness status.
 *
 * Uses text plus icon for contrast; color alone is insufficient.
 *
 * Properties:
 *   status: "current" | "stale" | "unknown"
 *   tooltip: optional longer description
 */
Control {
    id: badge
    required property string status
    property string tooltip: ""

    objectName: "freshnessBadge"

    Accessible.role: Accessible.Alert
    Accessible.name: _displayText + " " + status
    Accessible.description: tooltip

    padding: 4

    background: Rectangle {
        radius: 5
        color: {
            if (badge.status === "stale") return "#451A03"
            if (badge.status === "unknown") return "#1E293B"
            return "#052E16"
        }
        border.color: {
            if (badge.status === "stale") return "#F97316"
            if (badge.status === "unknown") return "#64748B"
            return "#22C55E"
        }
        border.width: 1
    }

    contentItem: RowLayout {
        spacing: 4

        Label {
            objectName: "freshnessIcon"
            text: {
                if (badge.status === "stale") return "⚠"
                if (badge.status === "unknown") return "?"
                return "✓"
            }
            color: {
                if (badge.status === "stale") return "#FB923C"
                if (badge.status === "unknown") return "#94A3B8"
                return "#4ADE80"
            }
            font.pixelSize: 12
            font.bold: true
        }

        Label {
            objectName: "freshnessText"
            text: _displayText
            color: {
                if (badge.status === "stale") return "#FDBA74"
                if (badge.status === "unknown") return "#CBD5E1"
                return "#86EFAC"
            }
            font.pixelSize: 12
        }
    }

    readonly property string _displayText: {
        if (badge.status === "stale") return "Stale"
        if (badge.status === "unknown") return "Unknown"
        return "Current"
    }
}
