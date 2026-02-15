# Blog Documentation

Hugo static site generator for https://junuxyz.github.io/

## Directory Structure

```
blog/
├── archetypes/          # Templates for new content
├── content/             # All blog posts and pages
├── layouts/             # HTML templates
├── static/              # CSS, JS, images
├── scripts/             # Build and utility scripts
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
│   ├── variables.css    # CSS variables and theme colors
│   ├── main.css         # Blog styles (all pages except homepage)
│   ├── home.css         # Homepage styles
│   └── syntax.css       # Code highlighting styles
├── js/                  # JavaScript
├── images/              # Images (use /images/ path in markdown)
└── CNAME                # GitHub Pages config
```

Active CSS files:
- `variables.css` - CSS variables (loaded first)
- `main.css` - all pages except homepage
- `home.css` - homepage only
- `syntax.css` - code highlighting




## Configuration (hugo.toml)

Key settings:
- `baseURL = "https://junuxyz.github.io/"` - site URL
- `title = "jlog"` - site name
- `pagerSize = 5` - posts per page
- Theme-less structure: all templates in `layouts/`, all assets in `static/`

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
math = true  # Enable KaTeX math rendering (optional)
+++
```

Required: `title`, `date`
Optional: `draft`, `categories`, `tags`, `description`, `math`

Set `draft = false` to publish. Posts won't appear on future dates.
Set `math = true` to enable KaTeX math rendering (only loads KaTeX libraries when needed).

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
- `static/css/variables.css` - CSS variables and theme colors (loaded first)
- `static/css/main.css` - All pages except homepage
- `static/css/home.css` - Homepage only
- `static/css/syntax.css` - Code highlighting

### How CSS Loading Works

`layouts/_default/baseof.html` conditionally loads CSS:

```html
<link rel="stylesheet" href="{{ "css/variables.css" | relURL }}">
{{ if .IsHome }}
  <link rel="stylesheet" href="{{ "css/home.css" | relURL }}">
{{ else }}
  <link rel="stylesheet" href="{{ "css/main.css" | relURL }}">
  <link rel="stylesheet" href="{{ "css/syntax.css" | relURL }}">
{{ end }}
```

### Edit Guidelines

**For CSS variables and theme colors**: Edit `static/css/variables.css`

**For blog post/page layout**: Edit `static/css/main.css`

**For homepage only**: Edit `static/css/home.css`

**For code highlighting**: Edit `static/css/syntax.css`

**Testing**: Changes auto-reload in `hugo server`. Use hard refresh (Shift+Reload) if cache persists.

## Math & Code

### Math Rendering (KaTeX)

Math rendering via KaTeX is **conditionally loaded** - only pages with `math: true` in front matter will load KaTeX libraries for better performance.

To enable math rendering in a post, add `math = true` to the front matter:

```toml
+++
title = "My Post with Math"
math = true
+++
```

Then use standard LaTeX syntax:
```
Display: $$E = mc^2$$
Inline: $E = mc^2$
```

**Performance Note**: Pages without `math: true` will not load KaTeX CSS/JS, reducing page load time.

### Code Blocks

Code blocks with syntax highlighting:
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

## Adding Images

### Using Obsidian

If you use Obsidian to write posts, configure it to work seamlessly with Hugo:

1. **Obsidian Settings** → **Files & Links**:
   - **Use Wiki links**: **Off** (필수) - Hugo Render Hook requires standard markdown links
   - **Default location for new attachments**: Choose one of:
     - `In subfolder under current folder` → `images/`
     - `Specified folder` → `static/images/`

2. **Image Path Format**: Use standard markdown image syntax:
   ```markdown
   ![Alt text](images/my-image.png)
   ```
   or
   ```markdown
   ![Alt text](/images/my-image.png)
   ```

3. **Hugo Render Hook**: The `layouts/_default/_markup/render-image.html` automatically converts:
   - `static/images/` → `/images/`
   - `images/` → `/images/`
   - Relative paths → `/images/`

### Image Location

Place images in `static/images/` directory. They will be served at `/images/` URL path.

**Example**:
- File location: `static/images/screenshot.png`
- Markdown: `![Screenshot](images/screenshot.png)` or `![Screenshot](/images/screenshot.png)`
- Rendered URL: `/images/screenshot.png`

## Deployment

The `public/` directory is deployed to GitHub Pages via GitHub Actions (`.github/workflows/deploy.yaml`).

**Note**: The deployment workflow no longer includes content processing steps (Obsidian file cleanup, path conversion) as these are handled by:
- `.gitignore` for Obsidian files
- Hugo Render Hook for image path conversion

## Common Issues

**Changes not showing**: Hard refresh (Shift+Reload) or restart `hugo server`

**Post not published**: Check `draft = false` and date is not in future

**CSS not working**: Verify file location. Root `static/css/` overrides theme CSS.

**Math not rendering**: 
- Ensure `math = true` is set in post front matter
- Only enabled on blog pages (not homepage)
- Check delimiters: `$$`, `$`, `\(`, `\[`
- Verify KaTeX libraries are loading in browser DevTools (Network tab)

**Templates broken**: Check file is in `layouts/` not theme. Verify Hugo syntax.

## Resources

- Hugo: https://gohugo.io/
- KaTeX: https://katex.org/
- Markdown: https://www.markdownguide.org/
