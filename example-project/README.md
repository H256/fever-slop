# FeverSlop Example Project

Copy this directory into `projects/`, then replace the input MP3 and edit the
`lyrics` and `music_style` fields in `config.json` for a new project. The
included track is only a ready-to-run example input.

## Run

Preview the planned work before starting expensive stages:

```powershell
uv run python main.py run ./example-project --dry-run
```

Run the project with:

```powershell
uv run python main.py run ./example-project --resume
```
