---
version: alpha
name: Descript
description: "A podcast and video editing platform built on a medium-dark canvas (#1C1C1E) with Descript's signature blue-violet primary (#5B5FC7 / #4F52B2) and white content panels for the transcript-based editing surface. The signature Descript insight is that audio/video editing should work like a document editor: the transcript is the timeline, and editing text edits the media. The UI reflects this: a clean document surface center-stage, media tracks below, and a dark chrome sidebar for project organization. Typography uses Inter for UI and a comfortable serif or sans for the transcript body to signal that this is a reading/writing environment as much as an editing one."

colors:
  primary: "#5B5FC7"
  on-primary: "#ffffff"
  primary-hover: "#4A4EB3"
  secondary: "#7B83EB"
  ink: "#1C1C1E"
  ink-muted: "#6E6E73"
  canvas: "#ffffff"
  surface-1: "#F5F5F7"
  surface-2: "#EBEBED"
  border: "#D8D8DC"
  editor-bg: "#1C1C1E"
  editor-panel: "#252528"
  editor-ink: "#E5E5EA"
  waveform: "#5B5FC7"
  filler-word: "#FFB800"
  deleted-text: "#FF453A"

typography:
  display:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: 44px
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: -0.02em
  body:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.7
    letterSpacing: 0
  transcript:
    fontFamily: "Georgia, Times New Roman, serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.75

spacing:
  base: 8px
  scale: [4, 8, 12, 16, 24, 32, 48, 64, 96]

radius:
  sm: 4px
  md: 8px
  lg: 12px
  pill: 9999px

shadows:
  card: "0 1px 4px rgba(0,0,0,0.12)"
  elevated: "0 4px 16px rgba(0,0,0,0.15)"
  editor-elevated: "0 4px 20px rgba(0,0,0,0.4)"

motion:
  duration-fast: 100ms
  duration-base: 200ms
  easing: cubic-bezier(0.4, 0, 0.2, 1)
---

## Rationale

**Georgia serif for transcript** — The choice to render transcripts in a serif (rather than the Inter used everywhere else) is the product's core metaphor made visible: this IS a document. When users see serif body text, their mental model shifts from "editing video" to "editing prose" — which is exactly how Descript wants them to think about their work.

**Filler word amber (#FFB800) as a productivity affordance** — Assigning a distinct, warm amber color to filler words makes a specific workflow (removing ums and uhs) so fast it feels like a superpower. The color is warm enough to be noticeable while editing but not alarming — it frames filler removal as improvement rather than error correction.

**Blue-violet (#5B5FC7) as the "waveform" color** — Tying the brand accent to the waveform and playhead creates a consistent metaphor: purple means "this is where the audio is." Users learn to find the playhead, find the speaker highlight, find the primary CTA all in the same color — reducing the number of distinct elements they need to track.

**Dark chrome + white document** — The split surface system (macOS dark editor shell, white document panel) mirrors how professional creative tools work (Final Cut, Logic) while communicating that the writing surface is privileged. It visually separates the production environment from the content, helping users stay in the right mental mode.

**Deleted text red strikethrough** — #FF453A on removed segments makes the "cutting by editing text" metaphor real: deleted words look like corrections on a printed draft. This feedback loop helps users trust that text deletion equals media deletion, which is Descript's core value proposition and biggest cognitive leap for new users.

## 1. Visual Theme & Atmosphere
Descript invented text-based video editing and the product design makes the metaphor real: you see a document, and the document IS the edit. Delete a sentence and the video skips that clip. The transcript surface is white and reading-comfortable, set at generous line-height because users need to read hours of content quickly. The purple-violet brand color appears on the waveform, playhead, and primary CTAs — providing the one chromatic thread in an otherwise neutral editing environment. Filler words (um, uh, like) are highlighted in amber for one-click removal.

## 2. Color System
**Editor (dark chrome + white document)**:
- Editor background: #1C1C1E — macOS dark system tone
- Document surface: white — the transcript needs to feel like paper
- Purple/violet: #5B5FC7 — waveform color, playhead, speaker label highlights
- Filler word highlight: #FFB800 — amber, immediately scannable for um/uh removal
- Deleted text: #FF453A — strikethrough red for removed segments

**Marketing/Web**: Standard light white canvas with purple primary

## 3. Typography
Georgia serif for transcript body — the document metaphor is reinforced typographically; serifs feel like reading, which is the right context for audio/video transcripts. Inter for all UI chrome (sidebar, toolbar, timeline). The separation is functional and meaningful.

## 4. Components & Patterns
- **Transcript editor**: Full-width document, speaker labels in color, filler words highlighted, playback cursor underlines current word
- **Timeline**: Below transcript, waveform tracks in purple, video thumbnail track, chapter markers
- **Screen recorder**: Floating HUD for capture (similar to Loom)
- **Overdub (AI voice)**: Underline styling on AI-generated text segments
- **Clip view**: Stacked word-by-word visual timeline for detailed trimming
- **Export panel**: Format picker, quality settings, upload destination options

## 5. Spacing & Layout
Transcript editor: 680px max content width, 24px horizontal padding. Timeline: full-width, ~120px fixed bottom. Project sidebar: 240px left. Right properties panel: 280px optional.

## 6. Motion & Interaction
Transcript playback underlines current word as audio plays — the core interaction. Filler word removal strikes through with animation. Timeline scrubbing is frame-perfect. AI regeneration shows a brief loading shimmer on the affected text.

## Accessibility

### Contrast Ratios
- **Primary on background** (#5B5FC7 on #FFFFFF): 5.4:1 — passes AA, fails AAA
- **Text on background** (#1C1C1E on #FFFFFF): 17.0:1 — passes AA, passes AAA
- **Muted on background** (#6E6E73 on #FFFFFF): 5.1:1 — passes AA, fails AAA

### Minimum Requirements
- **Touch target**: 44×44px minimum for all interactive elements
- **Focus indicator**: #5B5FC7 outline, 2px, 2px offset
- **Focus contrast**: 5.4:1 against #FFFFFF background

### Motion
- Respects `prefers-reduced-motion`: yes — all transitions and animations should be suppressed
- All transitions use `@media (prefers-reduced-motion: reduce)` guard

### Notes
- The blue-violet primary #5B5FC7 at 5.4:1 passes AA comfortably for normal text and is safe to use as a text link or interactive label on white.
- Transcript playback word-underline animation is the core interaction; under `prefers-reduced-motion` replace the animated underline with a static highlight that updates per word without transition.
- Filler-word strikethrough animation and AI regeneration shimmer should be replaced with instant state changes under `prefers-reduced-motion`.
- Muted #6E6E73 at 5.1:1 is safely above AA — suitable for captions and secondary metadata on white panels.
