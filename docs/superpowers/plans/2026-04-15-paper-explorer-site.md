# Paper Explorer Static Site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GitHub Pages static site that lets researchers filter and search all 466 mmWave radar papers by domain, year, artifact, and venue — without touching README.md.

**Architecture:** A Python script (`render_site.py`) reads the existing `papers.yml` + `taxonomy.yml` and writes `docs/papers.json`. A static `docs/index.html` fetches that JSON at runtime and renders a filterable, searchable list with expandable rows. All filtering is client-side; no build step; no new dependencies.

**Tech Stack:** Python 3 (stdlib + PyYAML, already a project dependency), vanilla JS (ES2020), CSS custom properties, GitHub Pages via `docs/` directory.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `.codex-local/auto-awesomemmwradar/scripts/render_site.py` | **Create** | Reads YAML data, writes `docs/papers.json` |
| `docs/index.html` | **Create** | Static app shell; fetches papers.json; all UI logic |
| `docs/papers.json` | **Generated** | Output of render_site.py; not hand-edited |
| `docs/mockup.html` | **Delete** | Replaced by index.html |

**Not touched:** `README.md`, `papers.yml`, `taxonomy.yml`, `publish_readme.py`, `run_daily_pipeline.py`.

---

## Context: Existing Script Pattern

All scripts in `.codex-local/auto-awesomemmwradar/scripts/` use this boilerplate:

```python
from __future__ import annotations
from pathlib import Path

try:
    from .catalog_tools import load_local_settings, load_yaml_list
except ImportError:
    from catalog_tools import load_local_settings, load_yaml_list

def main() -> None:
    settings = load_local_settings(Path(__file__))
    # settings.data_dir  → .codex-local/auto-awesomemmwradar/data/
    # settings.repo_root → D:\CodeXDev\Projects\awesome-mmwave-radar-perception\
    ...

if __name__ == "__main__":
    main()
```

`load_local_settings` reads `local_config.yml` (which lives in the `tool_root`) and returns a `LocalSettings` dataclass with `.repo_root`, `.data_dir`, `.reports_dir`, `.preview_path`, `.readme_path`.

Run scripts with the project venv:
```
D:\CodeXDev\Projects\awesome-mmwave-radar-perception\.codex-local\auto-awesomemmwradar\.venv\Scripts\python.exe <script.py>
```

---

## Task 1: render_site.py — Data Pipeline Script

**Files:**
- Create: `.codex-local/auto-awesomemmwradar/scripts/render_site.py`

- [ ] **Step 1: Create the script**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .catalog_tools import load_local_settings, load_yaml_list
except ImportError:
    from catalog_tools import load_local_settings, load_yaml_list


# ── Venue classification ──────────────────────────────────────────────────────
_CONF_KEYWORDS = {
    "CVPR", "NeurIPS", "ICCV", "ICRA", "IROS", "MobiCom", "MobiSys",
    "SenSys", "AAAI", "ECCV", "ICASSP", "ICLR", "NSDI", "SIGCOMM",
    "IMC", "CCS", "USENIX", "INFOCOM", "PerCom", "UbiComp", "CHI",
    "MobiHoc", "IPSN", "SenSys", "ICDM", "KDD", "WWW", "IROS",
    "RadarConf", "Asilomar", "MedComNet", "EGUGA", "BODYNETS",
}

def classify_venue(pub: str) -> str:
    """Return 'arxiv', 'journal', or 'conference'."""
    if not pub:
        return "conference"
    low = pub.lower()
    if "arxiv" in low:
        return "arxiv"
    up = pub.upper()
    if any(k in up for k in _CONF_KEYWORDS):
        return "conference"
    journal_markers = ["TRANSACTION", "JOURNAL", "LETTERS", "NATURE", "SCIENCE",
                       "PATTERN", "PATTERN", "DATA IN BRIEF", "REMOTE SENSING",
                       "SCIENTIFIC DATA", "SCIENTIFIC REPORTS", "SENSORS",
                       "ELECTRONICS", "PATTERNS", "INFORMATION FUSION",
                       "ICT EXPRESS", "GOOGLE BOOKS", "SSRN"]
    if any(m in up for m in journal_markers):
        return "journal"
    return "conference"


# ── Artifact normalisation ────────────────────────────────────────────────────
def normalize_artifact(label: str | None) -> str:
    """Normalise artifact_label to one of: Code, Dataset, Project, Baseline, N/A."""
    if not label or label in ("N/A", "TBD"):
        return "N/A"
    if "Code" in label:
        return "Code"
    if "Dataset" in label:
        return "Dataset"
    if label in ("Project", "Baseline"):
        return label
    return "N/A"


# ── Build taxonomy ────────────────────────────────────────────────────────────
def build_taxonomy(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a flat list of leaf topics with parent info."""
    by_id = {t["topic_id"]: t for t in topics}
    result = []
    for t in sorted(topics, key=lambda x: (x.get("sort_order", 99),)):
        if not t.get("is_leaf", True):
            continue
        parent_id = t.get("parent_id")
        parent = by_id.get(parent_id) if parent_id else None
        result.append({
            "topic_id": t["topic_id"],
            "label": t["title"],
            "parent_label": parent["title"] if parent else None,
            "parent_id": parent_id,
        })
    return result


# ── Build papers ──────────────────────────────────────────────────────────────
def build_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map papers.yml entries to the JSON schema expected by index.html."""
    result = []
    for p in papers:
        artifact_raw = p.get("artifact_label") or "N/A"
        artifact_norm = normalize_artifact(str(artifact_raw))
        artifact_url = p.get("artifact_url")
        # Ensure null URL for no-code papers
        if artifact_norm == "N/A":
            artifact_url = None

        result.append({
            "id": str(p["paper_id"]),
            "title": str(p.get("title", "")),
            "url": str(p.get("paper_url", "")),
            "publication": str(p.get("publication", "")),
            "year": int(p["year"]) if p.get("year") else None,
            "artifact_label": artifact_norm,
            "artifact_url": artifact_url or None,
            "topic_id": str(p.get("topic_id", "")),
            "summary": str(p.get("summary", "")),
        })
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def render_site(settings) -> dict[str, Any]:
    topics = load_yaml_list(settings.data_dir / "taxonomy.yml", "topics")
    papers = load_yaml_list(settings.data_dir / "papers.yml", "papers")

    payload = {
        "meta": {
            "total": len(papers),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "taxonomy": build_taxonomy(topics),
        "papers": build_papers(papers),
    }

    out_path = settings.repo_root / "docs" / "papers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "paper_count": len(papers),
        "topic_count": len(payload["taxonomy"]),
        "output_path": str(out_path),
    }


def main() -> None:
    settings = load_local_settings(Path(__file__))
    result = render_site(settings)
    print(f"Rendered {result['paper_count']} papers across {result['topic_count']} leaf topics.")
    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify JSON output**

```bash
cd /d/CodeXDev/Projects/awesome-mmwave-radar-perception/.codex-local/auto-awesomemmwradar
.venv/Scripts/python.exe scripts/render_site.py
```

Expected output (numbers may differ slightly):
```
Rendered 466 papers across 24 leaf topics.
Output: D:\CodeXDev\Projects\awesome-mmwave-radar-perception\docs\papers.json
```

- [ ] **Step 3: Spot-check papers.json structure**

```bash
python3 -c "
import json, sys
d = json.loads(open('docs/papers.json', encoding='utf-8').read())
print('meta:', d['meta'])
print('taxonomy[0]:', d['taxonomy'][0])
print('papers[0] keys:', list(d['papers'][0].keys()))
print('papers[0].artifact_label sample:', d['papers'][0]['artifact_label'])
# Check no 'Dataset || Code' survives normalisation
bad = [p for p in d['papers'] if '||' in str(p.get('artifact_label',''))]
print('unnormalised artifact rows:', len(bad))
" 
```

Expected: `unnormalised artifact rows: 0`, meta contains `total` ≥ 400, taxonomy entry has `topic_id`, `label`, `parent_label`, `parent_id`.

- [ ] **Step 4: Commit**

```bash
cd /d/CodeXDev/Projects/awesome-mmwave-radar-perception
git add .codex-local/auto-awesomemmwradar/scripts/render_site.py docs/papers.json
git commit -m "feat: add render_site.py and initial papers.json for GitHub Pages explorer"
```

---

## Task 2: docs/index.html — Production App Shell

**Files:**
- Create: `docs/index.html`
- Delete: `docs/mockup.html`

The HTML/CSS structure and all rendering/filter logic are identical to the approved mockup (`docs/mockup.html`). The two differences are:

1. Data is loaded via `fetch('papers.json')` instead of the hardcoded `const PAPERS = [...]` and `const TOPIC_META = {...}` arrays.
2. A loading state and error state are shown while fetch is in flight / if it fails.

- [ ] **Step 1: Create docs/index.html**

Write the file below in full. It is the mockup with hardcoded data replaced by fetch, a loading spinner, and an error fallback. All CSS variables, paper row rendering, filter sidebar, search, and sort logic are copied verbatim from `docs/mockup.html` with these surgical changes:

**Remove** from mockup.html:
- The `const PAPERS = [...]` block (22 sample papers, ~200 lines)
- The `const TOPIC_META = {...}` block (~30 lines)
- The `// ─── Init ───` two-liner at the bottom (`buildFilters(); render();`)
- The `.mockup-banner` CSS rule and its `<div class="mockup-banner">` HTML element

**Add/replace** the following:

In `<style>`, add after `::-webkit-scrollbar` rules:
```css
/* Loading / error states */
.state-overlay {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  color: var(--text-secondary);
  font-size: 14px;
  gap: 12px;
}
.state-overlay .icon { font-size: 32px; }
.spinner {
  width: 28px; height: 28px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
```

In `<main>`, replace the paper list + no-results block with:
```html
<div id="loading-state" class="state-overlay">
  <div class="spinner"></div>
  <span>Loading papers…</span>
</div>
<div id="error-state" class="state-overlay" style="display:none">
  <span class="icon">⚠️</span>
  <span id="error-msg">Failed to load papers.</span>
</div>
<div class="paper-list" id="paper-list" style="display:none"></div>
<div class="no-results" id="no-results" style="display:none">
  <div class="no-results-icon">🔍</div>
  <div class="no-results-text">No papers match your filters.<br>Try adjusting your search or clearing some filters.</div>
</div>
```

At the top of `<script>`, replace the hardcoded data blocks with:
```js
let PAPERS = [];
let TOPIC_META = {};

function buildTopicMeta(taxonomy) {
  TOPIC_META = {};
  for (const t of taxonomy) {
    TOPIC_META[t.topic_id] = { label: t.label, parent: t.parent_label };
  }
}
```

At the bottom of `<script>`, replace `buildFilters(); render();` with:
```js
async function init() {
  try {
    const resp = await fetch('papers.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status} — could not load papers.json`);
    const data = await resp.json();
    PAPERS = data.papers;
    buildTopicMeta(data.taxonomy);
    document.getElementById('total-badge').textContent = `${data.meta.total} papers`;
    document.getElementById('loading-state').style.display = 'none';
    document.getElementById('paper-list').style.display = 'flex';
    buildFilters();
    render();
  } catch (err) {
    document.getElementById('loading-state').style.display = 'none';
    document.getElementById('error-state').style.display = 'flex';
    document.getElementById('error-msg').textContent = err.message;
  }
}

init();
```

- [ ] **Step 2: Delete mockup.html**

```bash
rm /d/CodeXDev/Projects/awesome-mmwave-radar-perception/docs/mockup.html
```

- [ ] **Step 3: Verify index.html loads correctly via local server**

Because `fetch('papers.json')` uses a relative URL, open via a local HTTP server (not `file://`):

```bash
cd /d/CodeXDev/Projects/awesome-mmwave-radar-perception/docs
python3 -m http.server 8787
```

Open `http://localhost:8787` in the browser. Verify:
- Loading spinner appears briefly, then paper list renders
- Total count in header badge matches `papers.json` meta.total
- All four filter sections (Domain / Year / Artifact / Venue) populate with counts
- Typing in search box filters results in real-time
- Clicking a paper row expands the summary; clicking again collapses
- Clicking an artifact badge link opens the URL in a new tab without toggling the row
- Clicking a filter chip (×) removes that individual filter
- "↺ Reset filters" clears all state
- Sort dropdown reorders results

- [ ] **Step 4: Commit**

```bash
cd /d/CodeXDev/Projects/awesome-mmwave-radar-perception
git add docs/index.html
git rm docs/mockup.html
git commit -m "feat: add production Paper Explorer site (docs/index.html)"
```

---

## Task 3: GitHub Pages Configuration

**Files:**
- Modify: GitHub repository settings (web UI, not a file change)
- Create: `docs/.nojekyll` (prevents Jekyll processing)

- [ ] **Step 1: Create .nojekyll to disable Jekyll processing**

```bash
touch /d/CodeXDev/Projects/awesome-mmwave-radar-perception/docs/.nojekyll
```

This is required because GitHub Pages runs Jekyll by default, which can interfere with files starting with `_` or with JSON files.

- [ ] **Step 2: Commit .nojekyll**

```bash
cd /d/CodeXDev/Projects/awesome-mmwave-radar-perception
git add docs/.nojekyll
git commit -m "chore: add .nojekyll for GitHub Pages docs/ source"
```

- [ ] **Step 3: Enable GitHub Pages in repository settings**

Go to: `https://github.com/<owner>/awesome-mmwave-radar-perception/settings/pages`

Under **Source**, select:
- Branch: `main`
- Folder: `/docs`

Click **Save**. GitHub will build and deploy; the site URL will be shown (typically `https://<owner>.github.io/awesome-mmwave-radar-perception/`).

- [ ] **Step 4: Push all commits and verify live site**

```bash
cd /d/CodeXDev/Projects/awesome-mmwave-radar-perception
git push origin main
```

Wait ~60 seconds for GitHub Pages build, then open the Pages URL and confirm the site loads with real data.

---

## Task 4: Document the Publish Workflow

**Files:**
- No code changes. This task documents how `render_site.py` fits into the existing publish workflow.

The publish workflow is now two steps (run in order after new papers are added to `papers.yml`):

```bash
VENV=".codex-local/auto-awesomemmwradar/.venv/Scripts/python.exe"

# Step 1: Publish README.md (existing)
$VENV .codex-local/auto-awesomemmwradar/scripts/publish_readme.py

# Step 2: Publish papers.json for the web explorer (new)
$VENV .codex-local/auto-awesomemmwradar/scripts/render_site.py
```

- [ ] **Step 1: Update AGENTS.md to document the new step**

Open `AGENTS.md` and locate the existing publish workflow documentation. Add after the `publish_readme.py` step:

```markdown
## 发布网站数据（第二输出）

每次 `publish_readme.py` 执行后，运行：

```
.venv\Scripts\python.exe scripts\render_site.py
```

这会将 `papers.yml` + `taxonomy.yml` 渲染为 `docs/papers.json`，供 GitHub Pages 静态网站读取。`docs/index.html` 是静态文件，无需重新生成。
```

- [ ] **Step 2: Commit AGENTS.md update**

```bash
cd /d/CodeXDev/Projects/awesome-mmwave-radar-perception
git add AGENTS.md
git commit -m "docs: document render_site.py as second publish step"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| papers.json schema (meta, taxonomy, papers fields) | Task 1 Step 1 `build_taxonomy()` + `build_papers()` |
| Venue classification (arxiv / conference / journal) | Task 1 `classify_venue()` — note: classify_venue is defined but `build_papers()` does not output `venue_type`; venue classification runs in the browser JS using the same logic. This is correct — the JSON stores `publication` string; the browser classifies. No gap. |
| Artifact normalisation ("Dataset \|\| Code" → "Code") | Task 1 `normalize_artifact()` |
| Static HTML loads papers.json via fetch | Task 2 `init()` function |
| Loading state + error state | Task 2 Step 1 — spinner + error overlay |
| Filter: Domain (hierarchical, counts, ≥1 paper only) | Inherited from mockup; `buildFilters()` unchanged |
| Filter: Year descending with counts | Inherited from mockup |
| Filter: Artifact (Code/Dataset/Project/No artifact) | Inherited from mockup; normalisation now consistent via Task 1 |
| Filter: Venue (Conference/Journal/arXiv) | Inherited from mockup |
| Filter logic: AND across, OR within | Inherited from mockup `getFiltered()` |
| Active filter chips (per-chip remove) | Inherited from mockup `render()` chip section |
| Expandable rows (chevron, persist through filter change) | Inherited from mockup `toggleRow()` |
| Artifact badges: colour by type, link opens new tab | Inherited from mockup |
| Venue display: short name, full name in title attr | Inherited from mockup `shortVenue()` |
| Reset button | Inherited from mockup `resetFilters()` |
| Sort: newest / oldest / title | Inherited from mockup `sort-select` |
| docs/.nojekyll | Task 3 Step 1 |
| GitHub Pages via docs/ | Task 3 Step 3 |
| AGENTS.md workflow update | Task 4 |
| mockup.html removed | Task 2 Step 2 |

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:** `PAPERS` and `TOPIC_META` are declared as `let` at the top of the script block in index.html, matching their use in `buildFilters()`, `getFiltered()`, and `render()`. `buildTopicMeta()` populates `TOPIC_META` before `buildFilters()` is called. No mismatches.
