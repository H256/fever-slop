"""Stage-runner modules for the composition pipeline.

Each module owns one stage group. The facade module
``feverslop.composition.stage_runners`` re-exports everything so existing
import and ``patch("...stage_runners.X")`` targets keep working.
"""
