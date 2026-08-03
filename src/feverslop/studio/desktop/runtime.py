from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtQml import QQmlApplicationEngine

from feverslop.studio.desktop.composition import create_studio_context
from feverslop.studio.desktop.viewmodels.references import ReferenceWorkspaceViewModel
from feverslop.studio.desktop.viewmodels.rebuild import RebuildViewModel
from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel
from feverslop.studio.desktop.viewmodels.studio import StudioViewModel
from feverslop.studio.desktop.viewmodels.timeline import TimelineStudioViewModel
from feverslop.studio.job_service import thumbnail_path
from feverslop.studio.projects import ProjectStore
from feverslop.studio.reference_workspace_service import ReferenceWorkspaceService


def qml_entrypoint() -> QUrl:
    return QUrl.fromLocalFile(str(Path(__file__).with_name("qml") / "Main.qml"))


def _detect_color_mode() -> str:
    """Return 'dark' or 'light' based on system hints or env override."""
    override = os.environ.get("FEVSLOP_COLOR_MODE", "").lower()
    if override in ("dark", "light"):
        return override
    hints = QGuiApplication.styleHints()
    if hints is not None:
        scheme = hints.colorScheme()
        return "dark" if scheme == QPalette.ColorScheme.Dark else "light"
    return "light"


def _build_theme(color_mode: str) -> dict:
    """Build color tokens for QML theme context property."""
    is_dark = color_mode == "dark"
    if is_dark:
        return {
            "mode": "dark",
            "window": "#1C1C1E",
            "contentBg": "#1C1C1E",
            "contentHeaderBg": "#2C2C2E",
            "contentHeaderBgScene": "#202024",
            "contentHeaderText": "#F5F5F7",
            "contentHeaderSceneText": "#F4F4F5",
            "contentHeaderSeparator": "#3A3A3C",
            "contentHeaderSceneSeparator": "#3F3F46",
            "primaryText": "#F5F5F7",
            "secondaryText": "#8E8E93",
            "tertiaryText": "#6E6E73",
            "cardBg": "#2C2C2E",
            "cardBorder": "#3A3A3C",
            "inputBg": "#3A3A3C",
            "inputBorder": "#3F3F46",
            "navUnchecked": "#C7C7CC",
            "navUnhovered": "transparent",
            "navHovered": "#303034",
            "navChecked": "#3A3A40",
            "disabledItemBg": "#2A2A2D",
        }
    return {
        "mode": "light",
        "window": "#F5F5F7",
        "contentBg": "#F5F5F7",
        "contentHeaderBg": "#FFFFFF",
        "contentHeaderBgScene": "#202024",
        "contentHeaderText": "#1C1C1E",
        "contentHeaderSceneText": "#F4F4F5",
        "contentHeaderSeparator": "#D8D8DC",
        "contentHeaderSceneSeparator": "#3F3F46",
        "primaryText": "#1C1C1E",
        "secondaryText": "#6E6E73",
        "tertiaryText": "#8E8E93",
        "cardBg": "#FFFFFF",
        "cardBorder": "#D8D8DC",
        "inputBg": "#F5F5F7",
        "inputBorder": "#D8D8DC",
        "navUnchecked": "#C7C7CC",
        "navUnhovered": "transparent",
        "navHovered": "#303034",
        "navChecked": "#3A3A40",
        "disabledItemBg": "#EEEEF0",
    }


def studio_palette(color_mode: str = "light") -> QPalette:
    light_colors = {
        QPalette.ColorRole.Window: "#F5F5F7",
        QPalette.ColorRole.WindowText: "#1C1C1E",
        QPalette.ColorRole.Base: "#FFFFFF",
        QPalette.ColorRole.AlternateBase: "#EBEBED",
        QPalette.ColorRole.Text: "#1C1C1E",
        QPalette.ColorRole.Button: "#F5F5F7",
        QPalette.ColorRole.ButtonText: "#1C1C1E",
        QPalette.ColorRole.Highlight: "#5B5FC7",
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.PlaceholderText: "#6E6E73",
    }
    dark_colors = {
        QPalette.ColorRole.Window: "#1C1C1E",
        QPalette.ColorRole.WindowText: "#F5F5F7",
        QPalette.ColorRole.Base: "#2C2C2E",
        QPalette.ColorRole.AlternateBase: "#3A3A3C",
        QPalette.ColorRole.Text: "#F5F5F7",
        QPalette.ColorRole.Button: "#2C2C2E",
        QPalette.ColorRole.ButtonText: "#F5F5F7",
        QPalette.ColorRole.Highlight: "#5B5FC7",
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.PlaceholderText: "#8E8E93",
    }
    colors = dark_colors if color_mode == "dark" else light_colors
    palette = QPalette()
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    disabled_text = "#6E6E73" if color_mode == "light" else "#8E8E93"
    disabled_btn = "#8E8E93" if color_mode == "light" else "#6E6E73"
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(disabled_text))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(disabled_btn))
    return palette


def scene_video_thumbnail_url(
    store: ProjectStore,
    project_id: str,
    video_path: str,
) -> str:
    try:
        preview = thumbnail_path(store, project_id, video_path, 0.1)
    except (OSError, ValueError):
        return ""
    return QUrl.fromLocalFile(str(preview)).toString()


def run_studio(projects_root: str | Path, *, smoke_test: bool = False) -> int:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    app = QGuiApplication.instance() or QGuiApplication([])
    app.setApplicationName("FeverSlop Studio")
    app.setOrganizationName("FeverSlop")
    color_mode = _detect_color_mode()
    app.setPalette(studio_palette(color_mode))

    context = create_studio_context(projects_root)
    app.aboutToQuit.connect(
        lambda: context.jobs.shutdown(wait=False, cancel_futures=True)
    )
    view_model = StudioViewModel(
        store=context.store,
        jobs=context.jobs,
        job_service=context.job_service,
    )
    scene_view_model = SceneWorkspaceViewModel(
        service=context.scene_service,
        studio_view_model=view_model,
        thumbnail_url=lambda project_id, path: QUrl.fromLocalFile(
            str(context.store.resolve_media_path(project_id, path))
        ).toString(),
        video_thumbnail_url=lambda project_id, path: scene_video_thumbnail_url(
            context.store,
            project_id,
            path,
        ),
    )
    timeline_view_model = TimelineStudioViewModel(
        service=context.timeline_service,
        studio_view_model=view_model,
    )
    ref_service = ReferenceWorkspaceService(
        project_root=context.store.projects_root,
    )
    ref_view_model = ReferenceWorkspaceViewModel(
        service=ref_service,
        project_root=str(context.store.projects_root),
    )
    rebuild_view_model = RebuildViewModel(service=context.rebuild_service)
    view_model.currentProjectChanged.connect(lambda: rebuild_view_model.set_project_id(view_model.current_project_id))  # type: ignore[attr-defined]
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("studioViewModel", view_model)
    engine.rootContext().setContextProperty("sceneWorkspaceViewModel", scene_view_model)
    engine.rootContext().setContextProperty("timelineViewModel", timeline_view_model)
    engine.rootContext().setContextProperty("referenceWorkspaceViewModel", ref_view_model)
    engine.rootContext().setContextProperty("rebuildViewModel", rebuild_view_model)
    engine.rootContext().setContextProperty("theme", _build_theme(color_mode))
    engine._feverslop_view_models = (view_model, scene_view_model, timeline_view_model, ref_view_model, rebuild_view_model)  # type: ignore[attr-defined]
    engine.load(qml_entrypoint())
    if not engine.rootObjects():
        return 1
    if smoke_test:
        return 0
    view_model.refresh_projects()
    view_model.start_polling()
    return app.exec()
