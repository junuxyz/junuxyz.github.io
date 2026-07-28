# Site documentation

`junupark.xyz` is a theme-less Hugo site deployed to GitHub Pages.

## Canonical routes

| Route | Source | Purpose |
| --- | --- | --- |
| `/` | `content/_index.md` | Main page |
| `/projects/` | `content/projects/` + `data/projects.toml` | Project portfolio |
| `/blog/` | `content/blog/` | Non-technical essays |
| `/llm-engine/` | `llm-engine/` | Interactive LLM inference engine explainer |

Legacy technical blog posts and standalone pages are preserved under
`archive/`, which Hugo does not publish.

## Local development

Start Hugo with `hugo server`.

For a clean production build:

```bash
hugo --gc --minify --cleanDestinationDir
```

Run the interactive explainer independently:

```bash
cd llm-engine
npm ci
npm run dev
```

The deployment workflow builds the Vite app and copies its output into
`public/llm-engine/` after Hugo finishes.

## Editing

- Home copy: `content/_index.md`
- Projects: `data/projects.toml`
- Essays: `content/blog/*.md`
- ML systems notes: [`junuxyz/mlsys-notes`](https://github.com/junuxyz/mlsys-notes)
- Interactive explainer: `llm-engine/src/`
- Shared styling: `static/css/main.css`
- Home styling: `static/css/home.css`
