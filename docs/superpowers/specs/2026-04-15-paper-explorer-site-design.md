# Design Spec: Paper Explorer Static Site

**Date:** 2026-04-15  
**Status:** Approved  
**Author:** Armor (via brainstorming session)

---

## Goal

Add a second output format for the `awesome-mmwave-radar-perception` repository: a static GitHub Pages site that allows researchers to **precisely filter and search** the 466-paper curated list. The existing `README.md` output is unchanged; this site is an additive render target from the same `papers.yml` source.

---

## Core Decisions

| Dimension | Decision | Reason |
|---|---|---|
| Primary use case | Precise filtering (not browsing) | User decision |
| Visual style | Academic minimal (Papers With Code aesthetic) | User decision |
| Result display | Expandable rows (compact default, click to expand summary) | User decision |
| Architecture | Single HTML + papers.json, client-side JS | No build step; consistent with existing Python script pipeline |
| Hosting | GitHub Pages via `docs/` directory | Zero new infrastructure |

---

## Architecture

```
papers.yml  ──────────────────────────────────────┐
taxonomy.yml ─────────────────────────────────────┤
                                                   ▼
                                    render_site.py (new script)
                                                   │
                         ┌─────────────────────────┤
                         ▼                         ▼
                   docs/papers.json          docs/index.html
                   (data, ~150KB raw)        (app shell, static)
                                                   │
                                            fetch papers.json
                                            on page load
                                                   │
                                         client-side filter/search
```

**Data flow:**
1. `render_site.py` reads `papers.yml` + `taxonomy.yml`, outputs `docs/papers.json`
2. `docs/index.html` is a static app shell (no framework, vanilla JS)
3. On load, `index.html` fetches `papers.json` and renders the UI
4. All filtering and search happen client-side (no server)

**Deployment:** Adding `render_site.py` to the existing publish pipeline (after `publish_readme.py`). The `docs/` directory is configured as the GitHub Pages source in repo settings.

---

## UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│ 📡 mmWave Radar Perception  |  Paper Explorer    466 papers ↗README │
├──────────┬───────────────────────────────────────────────────┤
│          │  [🔍 Search titles and summaries…         ×]      │
│ DOMAIN   │                                                   │
│          │  Showing 23 papers · "doppler" × 2025 ×  ↕ Sort │
│ 🌐 Found.│  ─────────────────────────────────────────────── │
│   Signal │  ▶  DART: Implicit Doppler Tomography…  CVPR 2024 Code↗ │
│   SAR    │  ▼  Doppler Former: Velocity Supervision… ICASSP 2025 Code↗ │
│   Synth  │       Addresses the underutilization of velocity… │
│   FM     │       🚗 Foundational › Signal Processing   Read↗ │
│ 🚗 Auto  │  ─────────────────────────────────────────────── │
│   3D Det │  ▶  AdaRadar: Rate Adaptive Spectral…     arXiv 2026 Code↗ │
│   Odom   │                                                   │
│   Fusion │                                                   │
│   ...    │                                                   │
│ 🩺 Human │                                                   │
│   HAR    │                                                   │
│   Gest.  │                                                   │
│   ...    │                                                   │
│ 🌱 Agri  │                                                   │
│ 🏭 Indus │                                                   │
│ 🔒 Priv. │                                                   │
│ 📦 Other │                                                   │
│          │                                                   │
│ YEAR     │                                                   │
│ □ 2026 90│                                                   │
│ □ 2025 219│                                                  │
│ □ 2024 116│                                                  │
│ □ ...    │                                                   │
│          │                                                   │
│ ARTIFACT │                                                   │
│ □ Code 305│                                                  │
│ □ Dataset │                                                  │
│ □ Project │                                                  │
│ □ No code │                                                  │
│          │                                                   │
│ VENUE    │                                                   │
│ □ Conference│                                                │
│ □ Journal │                                                  │
│ □ arXiv  │                                                   │
│          │                                                   │
│ [↺ Reset]│                                                   │
└──────────┴───────────────────────────────────────────────────┘
```

---

## Component Specification

### Header
- Logo emoji + title + separator + subtitle
- Right: total paper count badge + "↗ View README" link
- Sticky, 52px height, white background, bottom border

### Sidebar (228px, sticky)
Four filter sections separated by subtle dividers:

1. **Domain** — hierarchical: parent group labels (non-selectable), indented topic checkboxes with paper counts. Only shows topics with ≥1 paper.
2. **Year** — checkboxes descending (2026 → 2019), with counts
3. **Artifact** — Code / Dataset / Project / No artifact, with counts
4. **Venue** — Conference / Journal / arXiv (preprint), with counts

Venue classification logic:
- `arxiv` → publication contains "arxiv" (case-insensitive)
- `conference` → publication matches any of: CVPR, NeurIPS, ICCV, ICRA, IROS, MobiCom, MobiSys, SenSys, AAAI, ECCV, ICASSP, ICLR, NSDI, SIGCOMM, IMC, CCS, USENIX, and others
- `journal` → publication contains "Transaction", "Journal", "Letters", "Nature", "Science"

Filter logic across categories: **AND** (intersection). Within a category: **OR** (union).

Reset button clears all filters + search.

### Search Bar
- Full-width, 14px, rounded-8
- Searches: `title` + `summary` fields (case-insensitive substring match)
- Clear (×) button appears when query is non-empty
- Focus ring: 3px rgba blue

### Stats Bar
- "Showing N papers" with bold count
- Active filter chips (clickable to remove individual filters)
- Sort dropdown: Newest first / Oldest first / Title A–Z

### Paper Rows (expandable)
**Collapsed state:**
```
▶ [Title as link]                    [Venue badge] [Year] [Artifact badge]
```

**Expanded state (on click):**
```
▼ [Title as link]                    [Venue badge] [Year] [Artifact badge]
  [Summary paragraph]
  [Topic breadcrumb chip]  [Read paper ↗]
```

- Chevron rotates 90° on expand (CSS transition 0.18s)
- Expand/collapse toggles on row click; artifact link and title link do NOT trigger toggle
- Expanded rows persist through filter/sort changes (by paper ID)
- Slide-down animation on expand (keyframe, 0.15s)

### Artifact Badges
| Value | Style |
|---|---|
| Code | Green bg (#f0fdf4), green text (#15803d) |
| Dataset | Amber bg (#fffbeb), amber text (#b45309) |
| Project | Purple bg (#f5f3ff), purple text (#7c3aed) |
| N/A / TBD | Gray bg (#f3f4f6), gray text (#9ca3af), label "No code" |

Badges with URLs are `<a>` tags (open in new tab). No-code badge is `<span>`.

### Venue Display
Long publication names are mapped to short abbreviations (e.g., "IEEE Transactions on Mobile Computing" → "IEEE TMC"). Full name shown in `title` attribute. Max-width 150px with text-overflow ellipsis.

---

## Color Tokens

```css
--bg: #ffffff
--bg-subtle: #f9fafb
--border: #e5e7eb
--border-subtle: #f3f4f6
--text-primary: #111827
--text-secondary: #6b7280
--text-tertiary: #9ca3af
--accent: #2563eb
```

Font: `-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif`

---

## papers.json Schema

```json
{
  "meta": {
    "total": 466,
    "generated_at": "2026-04-15T00:00:00Z"
  },
  "taxonomy": [
    {
      "topic_id": "autonomous-driving-drone__3d-object-detection-classification",
      "label": "3D Object Detection & Classification",
      "parent_label": "🚗 Autonomous Driving",
      "parent_id": "autonomous-driving-drone"
    }
  ],
  "papers": [
    {
      "id": "arxiv:2603.17979",
      "title": "AdaRadar: ...",
      "url": "https://arxiv.org/abs/...",
      "publication": "arXiv",
      "year": 2026,
      "artifact_label": "Code",
      "artifact_url": "https://github.com/...",
      "topic_id": "radar-foundational-technologies__signal-processing-parameter-estimation",
      "summary": "..."
    }
  ]
}
```

---

## New Files

| File | Purpose |
|---|---|
| `docs/index.html` | Static app shell (production, loads papers.json) |
| `docs/papers.json` | Rendered data from papers.yml (generated, not hand-edited) |
| `.codex-local/auto-awesomemmwradar/scripts/render_site.py` | New render script |

**Not changed:**
- `README.md` — untouched
- `papers.yml` / `taxonomy.yml` — untouched
- `publish_readme.py` — untouched

---

## Integration with Publish Pipeline

`render_site.py` is added as an optional step after `publish_readme.py`. It can be run standalone or as part of the daily pipeline. Output goes to `docs/` which GitHub Pages serves.

```
python render_site.py
# Reads: data/papers.yml, data/taxonomy.yml
# Writes: ../../docs/papers.json
# (index.html is static — not regenerated by script)
```

---

## Out of Scope (v1)

- Pagination (all 466 papers loaded client-side; acceptable at this scale)
- Dark mode
- Mobile layout optimization
- Citation counts or impact metrics
- Cross-topic paper tagging (multi-topic assignment)
- Server-side search
