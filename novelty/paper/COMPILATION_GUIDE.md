# LaTeX Compilation Guide
## COLM 2026 Paper: "Agent Memory Below the Prompt"

**Last Updated**: 2026-02-04
**Paper Status**: READY FOR COMPILATION

---

## Prerequisites

### Required Software

1. **TeX Distribution** (one of):
   - MacTeX 2023+ (recommended for macOS)
   - TeX Live 2023+
   - MiKTeX (Windows)

2. **LaTeX Compiler**:
   - `pdflatex` (required)
   - `bibtex` (required for bibliography)

3. **Optional Tools**:
   - LaTeX editor (TeXShop, TeXstudio, VS Code with LaTeX Workshop)
   - PDF viewer (Preview, Skim, Adobe Reader)

### Verify Installation

```bash
# Check pdflatex
pdflatex --version
# Should show: pdfTeX 3.14159265-2.6-1.40.24 or later

# Check bibtex
bibtex --version
# Should show: BibTeX 0.99d or later
```

---

## Compilation Steps

### Method 1: Command Line (Recommended)

```bash
# Navigate to paper directory
cd /Users/dev_user/semantic/novelty/paper/

# Step 1: First LaTeX pass (processes document structure)
pdflatex semantic_colm2026.tex

# Step 2: BibTeX pass (processes bibliography)
bibtex semantic_colm2026

# Step 3: Second LaTeX pass (integrates citations)
pdflatex semantic_colm2026.tex

# Step 4: Third LaTeX pass (resolves cross-references)
pdflatex semantic_colm2026.tex

# Result: semantic_colm2026.pdf generated
```

**Expected output**: Final PDF should be 9-10 pages (8.75 pages main text + appendices).

### Method 2: One-Command Script

Create a compilation script:

```bash
#!/bin/bash
# save as: compile.sh

set -e  # Exit on error

echo "=== COLM 2026 Paper Compilation ==="
echo "Stage 1: First pdflatex pass..."
pdflatex -interaction=nonstopmode semantic_colm2026.tex > /dev/null

echo "Stage 2: BibTeX pass..."
bibtex semantic_colm2026 > /dev/null

echo "Stage 3: Second pdflatex pass..."
pdflatex -interaction=nonstopmode semantic_colm2026.tex > /dev/null

echo "Stage 4: Third pdflatex pass..."
pdflatex -interaction=nonstopmode semantic_colm2026.tex > /dev/null

echo "=== Compilation Complete ==="
echo "Output: semantic_colm2026.pdf"
ls -lh semantic_colm2026.pdf
```

Make executable and run:
```bash
chmod +x compile.sh
./compile.sh
```

### Method 3: LaTeX Editor

**TeXShop / TeXstudio**:
1. Open `semantic_colm2026.tex`
2. Set compiler to `pdfLaTeX`
3. Click "Typeset" or press Cmd+T (Mac) / F5 (Windows)
4. Editor will automatically run all passes

**VS Code with LaTeX Workshop**:
1. Open `semantic_colm2026.tex`
2. Save file (triggers auto-build)
3. Or use: Cmd+Shift+P → "LaTeX Workshop: Build LaTeX project"

---

## Troubleshooting

### Common Issues

#### Issue 1: "File not found: colm2026_conference.sty"

**Cause**: Missing style file
**Fix**:
```bash
# Verify all template files are present
ls -la colm2026_conference.sty colm2026_conference.bst math_commands.tex fancyhdr.sty natbib.sty

# If missing, copy from Template-master:
cp ../papers_archive/Template-master/colm2026_conference.sty .
cp ../papers_archive/Template-master/colm2026_conference.bst .
# ... etc
```

#### Issue 2: "Undefined control sequence" errors

**Cause**: Missing TikZ or pgfplots packages
**Fix**:
```bash
# MacTeX/TeX Live: Update packages
sudo tlmgr update --self
sudo tlmgr install pgfplots tikz

# MiKTeX: Open MiKTeX Console → Updates → Update Now
```

#### Issue 3: Bibliography not appearing

**Cause**: BibTeX not run or .bib file errors
**Fix**:
```bash
# Check for BibTeX errors
bibtex semantic_colm2026

# Look for errors in output
# If "I found no \citation commands", check \cite{} in .tex file
# If "I couldn't open database file", check semantic_colm2026.bib exists
```

#### Issue 4: Figures not rendering

**Cause**: TikZ compilation errors or missing figure files
**Fix**:
```bash
# Verify all figure files exist
ls -la figures/*.tex

# Check TikZ logs in .log file
grep "TikZ" semantic_colm2026.log

# Common fix: Ensure tikz and pgfplots packages loaded
```

#### Issue 5: "Overfull \hbox" warnings

**Cause**: Lines too wide for page margins
**Fix**: These are warnings, not errors. To fix:
1. Reword long sentences
2. Use `\linebreak` where appropriate
3. Adjust hyphenation with `\hyphenation{word-list}`

**To find overfull boxes**:
```bash
grep "Overfull" semantic_colm2026.log
```

#### Issue 6: Citations showing as "?" in output

**Cause**: Need additional LaTeX pass or BibTeX errors
**Fix**: Run the 4-step compilation sequence again
```bash
pdflatex semantic_colm2026.tex
bibtex semantic_colm2026
pdflatex semantic_colm2026.tex
pdflatex semantic_colm2026.tex
```

---

## Verification Checklist

After successful compilation, verify:

### Page Count
```bash
# Count pages in PDF
pdfinfo semantic_colm2026.pdf | grep Pages
# Should show: Pages: 9-13 (main text ≤9, appendices unlimited)
```

### File Size
```bash
ls -lh semantic_colm2026.pdf
# Typical size: 200-500 KB (with TikZ figures)
```

### Visual Inspection

Open `semantic_colm2026.pdf` and check:

**Page 1 (Title)**:
- [ ] Title displays correctly
- [ ] "Anonymous Authors" shown (submission mode)
- [ ] Abstract starts on page 1
- [ ] Abstract is single paragraph, 150-180 words

**Figures**:
- [ ] Figure 1 (Architecture) renders correctly
- [ ] Figure 2 (TTFT Scaling) shows 3 lines (Cold/Warm/Hot)
- [ ] Figure 3 (Staggered Arrivals) shows grouped bars
- [ ] Figure 4 (UMA Comparison) shows side-by-side comparison
- [ ] All figures have captions
- [ ] Captions use "Figure N." format (bold period)

**Tables**:
- [ ] Table 1 (TTFT scaling) shows 5 columns (1K-16K)
- [ ] Table 2 (Batched throughput) shows 2 rows (Sequential/Batched)
- [ ] Tables use booktabs style (no vertical lines)
- [ ] Numbers are right-aligned

**References**:
- [ ] Bibliography starts on new page
- [ ] 25-30+ references listed
- [ ] Author-year format (not numbered [1] style)
- [ ] No "?" citations in text
- [ ] All URLs formatted correctly

**Appendices**:
- [ ] Start after references
- [ ] Labeled A, B, C
- [ ] Content displays correctly

---

## Output Files

After successful compilation, you'll have:

| File | Purpose | Keep? |
|------|---------|-------|
| `semantic_colm2026.pdf` | **Final output** | ✓ Yes (submit this) |
| `semantic_colm2026.aux` | Auxiliary file | ✗ No (temporary) |
| `semantic_colm2026.log` | Compilation log | ✓ Yes (for debugging) |
| `semantic_colm2026.bbl` | Processed bibliography | ✗ No (temporary) |
| `semantic_colm2026.blg` | BibTeX log | ✓ Yes (for debugging) |
| `semantic_colm2026.out` | Hyperref outline | ✗ No (temporary) |
| `semantic_colm2026.toc` | Table of contents | ✗ No (if generated) |

**Clean temporary files**:
```bash
rm -f *.aux *.log *.bbl *.blg *.out *.toc
# Keep only: .tex, .bib, .pdf, and figures/
```

---

## Advanced Options

### Silent Compilation (No Output)

```bash
pdflatex -interaction=batchmode semantic_colm2026.tex
# Output only on error
```

### Draft Mode (Fast Compilation)

Add `\documentclass[draft]{article}` to see:
- Boxes where figures would be (faster compilation)
- Overfull hbox markers

### Final Mode (Camera-Ready)

After acceptance, change:
```latex
% In semantic_colm2026.tex:
% \usepackage[submission]{colm2026_conference}  % Anonymous
\usepackage{colm2026_conference}  % Camera-ready with authors
```

Then recompile.

---

## Performance Tips

### Speed Up Compilation

1. **Use SSD**: Compile on SSD, not network drive
2. **Close unnecessary apps**: LaTeX uses CPU intensively
3. **Cache TikZ externalization**:
   ```latex
   \usetikzlibrary{external}
   \tikzexternalize[prefix=tikz-cache/]
   ```

### Reduce File Size

If PDF is too large:
```bash
# Compress PDF
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=semantic_colm2026_compressed.pdf semantic_colm2026.pdf
```

---

## Submission Preparation

### Final Checks Before Submission

1. **Anonymity**:
   ```bash
   grep -i "author" semantic_colm2026.tex
   # Should only show "Anonymous Authors"
   grep -i "github.com" semantic_colm2026.tex
   # Should show [anonymized] or no identifying URLs
   ```

2. **Page count**:
   - Main text: ≤ 9 pages (check PDF page breaks)
   - References + Appendices: unlimited

3. **File naming**:
   - Rename to conference requirements
   - Typically: `COLM2026_submission_XXXX.pdf`

4. **Supplementary materials**:
   - Create `supplementary.zip` with:
     - Source code (if open-sourcing)
     - Additional benchmark data
     - Extended appendices (if needed)

---

## Getting Help

### LaTeX Errors

1. **Read the .log file**:
   ```bash
   tail -100 semantic_colm2026.log
   # Shows last 100 lines (where errors usually appear)
   ```

2. **Search TeX Stack Exchange**:
   - https://tex.stackexchange.com/
   - Search error message

3. **Minimal Working Example**:
   - If stuck, create minimal .tex file reproducing error
   - Post to TeX Stack Exchange

### COLM Template Issues

- Check official COLM 2026 template: https://colmweb.org/cfp.html
- Verify using correct year's template (2026, not 2025)

---

## Success Indicators

**Compilation successful if**:
- ✅ `semantic_colm2026.pdf` generated
- ✅ PDF opens without errors
- ✅ All figures render
- ✅ Bibliography appears
- ✅ Page count ≤ 9 pages main text
- ✅ No "?" in citations
- ✅ No missing figure boxes

**Next step**: Visual inspection (see PAPER_COMPLETE_SUMMARY.md checklist)

---

**END OF COMPILATION GUIDE**

For issues, check:
1. `semantic_colm2026.log` for LaTeX errors
2. `semantic_colm2026.blg` for BibTeX errors
3. `review/investigation.md` for internal consistency checks
