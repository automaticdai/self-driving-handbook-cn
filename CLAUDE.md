# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese-language documentation site about autonomous driving technology (自动驾驶技术指南). Built with MkDocs and the Material theme, deployed to GitHub Pages via GitHub Actions.

- **Site URL:** https://yfrobotics.github.io/self-driving-handbook-cn
- **License:** CC 4.0-BY-SA
- **Language:** All content is in Simplified Chinese (zh)

## Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Serve locally with live reload (primary dev command)
mkdocs serve

# Build static site to site/ directory (strict is enabled in mkdocs.yml;
# CI also runs this and fails the build on warnings such as broken links)
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Architecture

This is a pure documentation project — no application code, tests, or linting.

**Content structure** (`docs/`):
- `index.md` — Home page
- `intro/` — Section 1: Overview and definitions of autonomous driving
- `system/` — Section 2: Vehicle systems, V2X, platforms, safety, regulations
- `hardware/` — Section 3: CCU, drive-by-wire, sensors, cameras
- `algorithm/` — Section 4: Image/laser processing, localization, path planning, decision making, NLP
- `simulation/` — Section 5: Simulation and testing
- `vlm/` — Section 6: Vision-language models
- `casestudy/` — Section 7: Real-world examples (Apollo, Waymo, Tesla)

**Navigation** is defined explicitly in `mkdocs.yml` under the `nav` key. Adding a new page requires updating both the markdown file and the nav config.

**Math support:** MathJax 3 is loaded via CDN. Use `$...$` for inline math and `$$...$$` for display math. Configuration is in `docs/_static/js/mathjaxhelper.js`.

**Markdown extensions:** `admonition`, `pymdownx.arithmatex`, `pymdownx.superfences`, `pymdownx.snippets`, `tables`, `fenced_code`, `abbr`, `extra`.

**Diagrams:** Mermaid is enabled via a `pymdownx.superfences` custom fence — use a ```mermaid block. Material loads mermaid from a CDN at runtime, so diagrams render client-side. Validate syntax before committing with `npx -p @mermaid-js/mermaid-cli mmdc -i diagram.mmd -o out.png`; a syntax error renders as an error box on the page rather than failing the build.

**Abbreviations:** `includes/abbreviations.md` is auto-appended to every page, so terms listed there (TOPS, ASIL, ADAS, …) become `<abbr>` tooltips automatically. Do not redefine them per-page.

**Chinese search:** the `search` plugin depends on `jieba` (in `requirements.txt`) for Chinese word segmentation. jieba separates tokens with a zero-width space, so the `separator` regex in `mkdocs.yml` must keep `\u200b` — dropping it makes Chinese search fail almost entirely.

## CI/CD

GitHub Actions (`.github/workflows/main.yml`) runs on push/PR to `master`:

1. **build** — `pip install -r requirements.txt`, then `mkdocs build --strict`. This gates every PR, so a broken internal link or missing image fails CI.
2. **deploy** — only on push to `master`, and only if build passed: `mkdocs gh-deploy --force`.

Both jobs use `fetch-depth: 0` because the `git-revision-date-localized` plugin needs full history.
