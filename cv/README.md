# CV - Junu Park

This directory contains the CV in HTML format that can be converted to PDF.

## Generating PDF

### Method 1: Using Node.js + Puppeteer (Recommended for WSL)

If you have Node.js installed (which you do), install puppeteer and run:

```bash
cd cv
npm install puppeteer
./generate-pdf.sh
```

Or directly:
```bash
node generate-pdf.js
```

### Method 2: Using Browser (Windows)

Since you're on WSL, you can open the HTML file in Windows:

1. In Windows Explorer, navigate to: `\\wsl$\Ubuntu\home\jpotw\Projects\blog\cv\index.html`
   (Replace `Ubuntu` with your WSL distribution name if different)
2. Or use the full path: `\\wsl$\[your-distro-name]\home\jpotw\Projects\blog\cv\index.html`
3. Double-click to open in your default browser
4. Press `Ctrl+P` to open the print dialog
5. Select "Save as PDF" as the destination
6. Click "Save"

### Method 3: Install wkhtmltopdf

```bash
sudo apt install wkhtmltopdf
cd cv
./generate-pdf.sh
```

### Method 4: Using Script (Auto-detects tools)

Run the provided script:
```bash
cd cv
./generate-pdf.sh
```

The script will automatically detect and use available PDF generation tools:
- Node.js + puppeteer (if installed)
- `wkhtmltopdf` (if installed)
- Chromium/Chrome (if installed)
- Firefox (opens HTML for manual printing)

## File Structure
- `index.html` - CV content in HTML format
- `generate-pdf.sh` - Script to automate PDF generation
- `generate-pdf.js` - Node.js script using puppeteer
- `README.md` - This file

