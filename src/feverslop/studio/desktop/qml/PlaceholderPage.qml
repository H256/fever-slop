import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    required property string heading
    required property string detail

    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(560, parent.width - 56)
        spacing: 10
        Label { text: heading; color: "#1C1C1E"; font.bold: true; font.pixelSize: 22; Layout.alignment: Qt.AlignHCenter }
        Label { text: detail; color: "#6E6E73"; font.pixelSize: 15; wrapMode: Text.WordWrap; horizontalAlignment: Text.AlignHCenter; Layout.fillWidth: true }
    }
}
