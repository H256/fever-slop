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


def studio_palette() -> QPalette:
    palette = QPalette()
    colors = {
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
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6E6E73"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#8E8E93"))
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


def run_studio(projects_root: str | Path) -> int:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    app = QGuiApplication.instance() or QGuiApplication([])
    app.setApplicationName("FeverSlop Studio")
    app.setOrganizationName("FeverSlop")
    app.setPalette(studio_palette())

    context = create_studio_context(projects_root)
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
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("studioViewModel", view_model)
    engine.rootContext().setContextProperty("sceneWorkspaceViewModel", scene_view_model)
    engine.rootContext().setContextProperty("timelineViewModel", timeline_view_model)
    engine.rootContext().setContextProperty("referenceWorkspaceViewModel", ref_view_model)
    engine.rootContext().setContextProperty("rebuildViewModel", rebuild_view_model)
    engine._feverslop_view_models = (view_model, scene_view_model, timeline_view_model, ref_view_model, rebuild_view_model)  # type: ignore[attr-defined]
    engine.load(qml_entrypoint())
    if not engine.rootObjects():
        return 1
    view_model.refresh_projects()
    view_model.start_polling()
    return app.exec()
