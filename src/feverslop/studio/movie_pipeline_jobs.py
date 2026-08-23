"""Backward-compatible import location for movie pipeline jobs.

The implementation lives in :mod:`feverslop.application.movie_pipeline_jobs`
so CLI composition does not depend on the Studio package.
"""

from feverslop.composition.movie_pipeline_jobs import *  # noqa: F403
