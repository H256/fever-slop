from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from feverslop.studio.desktop.composition import create_studio_context
from feverslop.studio.desktop.viewmodels.studio import StudioViewModel


def qml_entrypoint() -> QUrl:
    return QUrl.fromLocalFile(str(Path(__file__).with_name("qml") / "Main.qml"))


def run_studio(projects_root: str | Path) -> int:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    app = QGuiApplication.instance() or QGuiApplication([])
    app.setApplicationName("FeverSlop Studio")
    app.setOrganizationName("FeverSlop")

    context = create_studio_context(projects_root)
    view_model = StudioViewModel(
        store=context.store,
        jobs=context.jobs,
        job_service=context.job_service,
    )
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("studioViewModel", view_model)
    engine.load(qml_entrypoint())
    if not engine.rootObjects():
        return 1
    view_model.refresh_projects()
    view_model.start_polling()
    return app.exec()
