# Blog Documentation

Hugo static site generator for https://junuxyz.github.io/

## Directory Structure

```
blog/
├── archetypes/          # Templates for new content
├── content/             # All blog posts and pages
├── layouts/             # HTML templates
├── static/              # CSS, JS, images
├── themes/              # Custom theme (active: custom/)
├── hugo.toml            # Config file
└── public/              # Build output
```

### content/

```
content/blog/
├── _index.md            # Blog index page
├── about.md             # About page
├── categories.md        # Categories archive
├── tags.md              # Tags archive
├── posts/               # Blog posts (this is where content goes)
├── categories/          # Category pages
└── tags/                # Tag pages
```

New posts go in `content/blog/posts/` as markdown files.

### layouts/

```
layouts/
├── _default/
│   ├── baseof.html      # Base template (all pages inherit from this)
│   ├── single.html      # Single post page
│   ├── list.html        # Post list/archive
│   ├── about.html       # About page
│   ├── categories.html  # Category pages
│   └── tags.html        # Tag pages
├── index.html           # Homepage
├── blog/                # Blog-specific overrides
└── partials/            # Reusable components (header, footer)
```

Project-level templates in `layouts/` override theme templates.

### static/

```
static/
├── css/
│   ├── main.css         # Blog styles (active)
│   ├── minimal.css      # Homepage styles (active)
│   └── main-scoped.css  # Unused
├── js/                  # JavaScript
├── images/              # Images
└── CNAME                # GitHub Pages config
```

Active CSS files:
- `main.css` - all pages except homepage
- `minimal.css` - homepage only
- Code highlighting: `themes/custom/static/css/syntax.css`



### themes/custom/

Active custom theme. `syntax.css` handles code highlighting. Theme files are overridden by same-path files in project root (`layouts/` and `static/`).

## Configuration (hugo.toml)

Key settings:
- `baseURL = "https://junuxyz.github.io/"` - site URL
- `title = "jlog"` - site name
- `theme = "custom"` - active theme
- `pagerSize = 5` - posts per page

Permalinks:
- Categories: `/blog/categories/:slug/`
- Tags: `/blog/tags/:slug/`
- Posts: `/blog/posts/:slug/`

Menu items are defined in `[menu.main]` section. Add/remove/reorder navigation here.

## Creating Posts

### New post using archetype (recommended)
```bash
hugo new content/blog/posts/my-post.md
```

Creates file with template from `archetypes/default.md`.

### Front matter format
```toml
+++
title = "Post Title"
date = 2025-11-03T10:00:00Z
draft = false
categories = ["Category"]
tags = ["tag1", "tag2"]
+++
```

Required: `title`, `date`
Optional: `draft`, `categories`, `tags`, `description`

Set `draft = false` to publish. Posts won't appear on future dates.

## Running Locally

```bash
hugo server
# http://localhost:1313
```

## Building

```bash
hugo
# Generates public/ directory
```

## Modifying Styles

### CSS Files

Active CSS (modify these):
- `static/css/main.css` - All pages except homepage
- `static/css/minimal.css` - Homepage only
- `themes/custom/static/css/syntax.css` - Code highlighting

Inactive (ignore):
- `themes/custom/static/css/main.css` - Overridden by root `static/css/main.css`
- `static/css/main-scoped.css` - Empty placeholder, unused

### How CSS Loading Works

`layouts/_default/baseof.html` conditionally loads CSS:

```html
{{ if .IsHome }}
  <link rel="stylesheet" href="{{ "css/minimal.css" | relURL }}">
{{ else }}
  <link rel="stylesheet" href="{{ "css/main.css" | relURL }}">
  <link rel="stylesheet" href="{{ "css/syntax.css" | relURL }}">
{{ end }}
```

Project root files take precedence over theme files with the same path.

### Edit Guidelines

**For blog post/page layout**: Edit `static/css/main.css`

**For homepage only**: Edit `static/css/minimal.css`

**For code highlighting**: Edit `themes/custom/static/css/syntax.css` or copy to `static/css/syntax.css` for project-level management

**Testing**: Changes auto-reload in `hugo server`. Use hard refresh (Shift+Reload) if cache persists.

## Math & Code

Math rendering via KaTeX (already configured):
```
Display: $$E = mc^2$$
Inline: $E = mc^2$
```

Code blocks:
````markdown
```python
print("hello")
```
````

## Adding Features

### New category

Create `content/blog/categories/my-category/_index.md`:
```toml
+++
title = "My Category"
+++
```

Add to menu in `hugo.toml`:
```toml
[[menu.main]]
  identifier = "my-category"
  name = "My Category"
  url = "/blog/categories/my-category/"
  weight = 25
```

Tag posts with `categories = ["My Category"]`.

### New page

Create `content/blog/my-page.md`. Optional: create custom template at `layouts/_default/my-page.html`.

### Custom JavaScript

Add `.js` file to `static/js/`, then load in `layouts/_default/baseof.html`:
```html
<script src="{{ "js/my-script.js" | relURL }}"></script>
```

## Deployment

The `public/` directory is deployed to GitHub Pages. Ensure GitHub Actions or your deployment script is configured.

## Common Issues

**Changes not showing**: Hard refresh (Shift+Reload) or restart `hugo server`

**Post not published**: Check `draft = false` and date is not in future

**CSS not working**: Verify file location. Root `static/css/` overrides theme CSS.

**Math not rendering**: Only enabled on blog pages (not homepage). Check delimiters: `$$`, `$`, `\(`, `\[`.

**Templates broken**: Check file is in `layouts/` not theme. Verify Hugo syntax.

## Resources

- Hugo: https://gohugo.io/
- KaTeX: https://katex.org/
- Markdown: https://www.markdownguide.org/
