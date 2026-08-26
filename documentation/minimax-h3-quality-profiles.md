# MiniMax H3 quality profiles

Alle H3-Profile verwenden zwei Passes; ein dritter Pass ist nicht vorgesehen.
`draft` bleibt der Default für neue Projekte.

| Profil | Pass 1 Steps | Pass 2 Steps | Pass 2 Denoise | Zweck |
|---|---:|---:|---:|---|
| draft | 12 | 4 | 0.55 | schnelle Planung und Tests |
| standard | 20 | 8 | 0.40 | ausgewogene Produktion |
| final | 28 | 12 | 0.30 | maximale Detailverfeinerung |

Die Werte sind reproduzierbare Startkalibrierungen, keine hardwareunabhängigen
Laufzeit- oder VRAM-Versprechen. Die genaue Auflösung wird separat über das
jeweilige Auflösungsprofil bestimmt. Bei Audio-Profilen bleibt der Audio-Latent
Zweig unverändert und darf keinen räumlichen Upscaler durchlaufen.
