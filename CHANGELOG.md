# Changelog

## 1.1.0 - 2026-09-17

Library phrases are spoken lines now, and they are used.

- **Speech rule for the library phrases.** A stretch of the vocals stem becomes a phrase when it behaves like
  talking: gaps between syllables, consonants alternating with vowels, energy in the voice band, a tonal
  spectrum, no pulsing locked to the beat. Sentences stay whole (pauses up to 0.7 s are bridged), 1.2-8 s.
  Loudest-first picked pads, chops and effects; on a 718-track library the new rule keeps 1,470 phrases and
  leaves 139 tracks with none, which is the point.
- **Library > Re-extract phrases** runs the rule again over every analysed track (about 16 s each on an RTX
  3060), replaces the old picks (files go to `data/trash`) and is resumable. Each track remembers the settings
  its phrases came from, so a changed minimum speech score reruns exactly what changed.
- **Settings > Library analysis > Minimum speech score** (0.5).
- **Phrase bank**: the filter shows counts, a line says how much of the bank is on screen, the list fills the
  window, and the bank refreshes as phrases land. Library phrases are named after their track.
- **Compose > Which phrases** shows the count behind every source and defaults to *all phrases* (one-time
  settings migration from *captured*).
- Errors inside a button handler are logged to `data/logs/tmg.log` and shown as a toast instead of vanishing.
- Fixed: the Re-extract button did nothing (a property called as a method).

## 1.0.0 - 2026-09-15

First public release: library learning (Demucs, beats, patterns, sections, key), the style profile, the
composer with a timeline, phrases from the system audio with the trim editor, GLSL visuals with NVENC,
MusicGen and Stable Audio Open textures with A/B, the neural model self-test, install script, launcher and
icon, installation guide.
