# LTX Prompt Anchor Fix

Die Anchor-Fix-Dokumentation wurde in die Haupt-README konsolidiert.

Siehe:

- `README.md` Abschnitt `Empfohlene Befehlsreihenfolge`
- `README.md` Abschnitt `Optional: LTX Anchors fixen`
- `README.md` Abschnitt `Debug-Checkliste`

Der relevante CLI-Befehl bleibt:

```powershell
uv run python fix_ltx_prompt_anchors.py `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --subject-anchor "the old weary warrior man with weathered scarred face, salt-and-pepper beard, tattered leather armor, and heavy frayed cloak"
```
