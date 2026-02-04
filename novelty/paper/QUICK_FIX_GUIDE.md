# Quick Fix Guide: Diagrams

## 🔴 CRITICAL FIX: Figure 1 Text Overlap

### Problem
Figure 1 (UMA comparison) has severe text overlap in bandwidth annotations.

### Solution (2 minutes)

```bash
cd /Users/dev_user/semantic/novelty/paper/

# Backup original
cp figures/fig_uma_comparison.tex figures/fig_uma_comparison.tex.BACKUP

# Use fixed version
cp figures/fig_uma_comparison_FIXED.tex figures/fig_uma_comparison.tex

# Recompile
eval "$(/usr/libexec/path_helper)" && \
pdflatex -interaction=nonstopmode semantic_colm2026.tex && \
pdflatex -interaction=nonstopmode semantic_colm2026.tex

# Check result
open semantic_colm2026.pdf
```

### What Changed

**Before (BROKEN):**
```latex
% Annotations positioned relative to TITLES at top
\node[below=1.5cm of discrete_title] { A100: 1,555 GB/s ... };
\node[below=1.5cm of uma_title] { M4 Pro: 273 GB/s ... };
% Problem: Different absolute Y-coordinates → OVERLAP
```

**After (FIXED):**
```latex
% Annotations positioned relative to BOTTOM components
\node[below=1.0cm of gpu_compute] (left_bandwidth) { A100 VRAM: 1,555 GB/s ... };
\node[below=1.0cm of cpu_gpu] (right_bandwidth) { M4 Pro: 273 GB/s ... };
% Convergence note positioned relative to annotation (not title)
\node[below=0.5cm of right_bandwidth] { Bandwidth convergence ... };
```

---

## 🟡 OPTIONAL: Figure 2 Polish

### Problem
Disk circle too large, bottom annotations cramped.

### Solution

Edit `figures/fig_architecture.tex`:

**Find this (around line 40):**
```latex
\node[circle, draw, fill=yellow!30] (disk) at (10,1) {
    Disk\\
    safetensors\\
    50ms reload
};
```

**Change to:**
```latex
\node[circle, draw, fill=yellow!30, minimum size=2cm, font=\small] (disk) at (10,1) {
    Disk\\
    safetensors\\
    50ms reload
};
```

**For bottom annotations (if they exist as absolute positioning):**

Replace absolute `\node at (x,y)` with relative positioning:
```latex
% INSTEAD OF: \node at (3,−1) {2.0--4.3× E2E speedup};
% USE:
\node[below=0.3cm of block_pool, font=\small, color=blue!70!black] {
    2.0--4.3$\times$ E2E speedup
};
```

---

## 🛠️ AUTOMATED TESTING

### Run Overlap Detection

```bash
# Install required package
pip install pdfplumber

# Run detector
python detect_diagram_issues.py \
    --pdf semantic_colm2026.pdf \
    --log semantic_colm2026.log \
    --tikz-dir figures/ \
    --output DIAGRAM_QUALITY_REPORT.txt

# View report
cat DIAGRAM_QUALITY_REPORT.txt
```

**Expected output after fix:**
```
### Page 3: Figure 1 & 2 (Memory Architecture + System Design)
  ✓ No text overlaps detected
```

---

## 🤖 CLAUDE API AUTOMATED REVIEW

### Setup

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Or add to env.json (already configured in this project)
```

### Python Script for Batch Review

```python
#!/usr/bin/env python3
import anthropic
import base64
from pathlib import Path

def review_tikz_with_claude(tikz_file: str, screenshot_path: str) -> str:
    """
    Use Claude to review TikZ code and suggest fixes
    """
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    # Read TikZ source
    tikz_code = Path(tikz_file).read_text()

    # Read screenshot (if available)
    if Path(screenshot_path).exists():
        with open(screenshot_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""Review this TikZ diagram for quality issues.

TikZ Source:
```latex
{tikz_code}
```

Check for:
1. Text overlaps or cramped spacing
2. Disproportionate elements
3. Absolute vs relative positioning
4. Visual clarity issues

Provide:
1. Specific problems found (with line numbers)
2. Corrected LaTeX code
3. Brief explanation of fixes

Be concise and focus on critical issues only."""
                    }
                ]
            }]
        )
    else:
        # Code-only review (no screenshot)
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": f"""Review this TikZ code for potential layout issues.

```latex
{tikz_code}
```

Check for:
- Absolute positioning (less flexible than relative)
- Unnamed nodes (harder to reference)
- Potential text overlap risks
- Mixed units (cm, pt, em)

Suggest specific improvements with corrected code."""
            }]
        )

    return message.content[0].text

# Example usage
if __name__ == "__main__":
    result = review_tikz_with_claude(
        tikz_file="figures/fig_uma_comparison.tex",
        screenshot_path="screenshots/fig_uma.png"  # Optional
    )
    print(result)
```

**Save as**: `tikz_review_claude.py`

**Run:**
```bash
python tikz_review_claude.py
```

---

## 📊 VERIFICATION CHECKLIST

After applying fixes:

```bash
# 1. Recompile PDF
pdflatex semantic_colm2026.tex && pdflatex semantic_colm2026.tex

# 2. Run automated tests
python detect_diagram_issues.py

# 3. Visual inspection
open semantic_colm2026.pdf

# Go to page 3 and verify:
# [ ] No text overlaps in Figure 1
# [ ] Bandwidth annotations clearly separated
# [ ] Convergence note properly positioned
# [ ] All text readable at 100% zoom

# 4. Check compilation warnings
grep -i "overfull" semantic_colm2026.log | wc -l
# Should be 0 or minimal

# 5. Check for LaTeX errors
grep "^!" semantic_colm2026.log
# Should be empty
```

---

## 🎯 SUCCESS CRITERIA

### Before Fix:
- ❌ Text overlap detected in Figure 1
- ❌ "A100: 1,555 GB/s" overlaps "PCIe 4.0: 32 GB/s"
- ❌ Convergence note positioning ambiguous
- **Quality: 3/10**

### After Fix:
- ✅ No text overlaps detected
- ✅ All bandwidth annotations clearly separated
- ✅ Convergence note positioned logically
- ✅ Relative positioning used throughout
- **Quality: 8/10** (publication ready)

---

## ⏱️ TIME ESTIMATES

- **Apply Figure 1 fix**: 2 minutes (copy + recompile)
- **Apply Figure 2 polish**: 5 minutes (edit + test)
- **Run automated tests**: 1 minute
- **Visual verification**: 3 minutes
- **Total**: ~11 minutes to fix all issues

---

## 🔗 RESOURCES

### Documentation
- `DIAGRAM_VISUAL_AUDIT.md` - Detailed analysis of all issues
- `detect_diagram_issues.py` - Automated detection script
- `figures/fig_uma_comparison_FIXED.tex` - Corrected Figure 1

### Online Tools
- **TikZiT**: https://tikzit.github.io/ (visual editor)
- **Overleaf**: Real-time compilation
- **pdfplumber docs**: https://github.com/jsvine/pdfplumber

### Papers on TikZ Best Practices
- "Beautiful figures in LaTeX" (2023) - Excellent TikZ patterns
- "Graphics for Communication" - Visual design principles
- PGF/TikZ manual: http://mirrors.ctan.org/graphics/pgf/base/doc/pgfmanual.pdf

---

## 🚨 TROUBLESHOOTING

### "No module named 'pdfplumber'"
```bash
pip install pdfplumber
```

### "TikZ figure doesn't render after changes"
```bash
# Clear auxiliary files
rm -f semantic_colm2026.aux semantic_colm2026.out
# Recompile from scratch
pdflatex semantic_colm2026.tex
pdflatex semantic_colm2026.tex
```

### "Changes don't appear in PDF"
```bash
# Make sure you're editing the correct file
ls -la figures/fig_uma_comparison.tex
# Check modification time
stat figures/fig_uma_comparison.tex
# Force recompile
touch semantic_colm2026.tex
pdflatex semantic_colm2026.tex
```

### "Still seeing text overlap"
```bash
# Verify you copied the fixed version
diff figures/fig_uma_comparison.tex figures/fig_uma_comparison_FIXED.tex
# Should show: Identical files (if fix was applied)
```

---

**Last Updated**: 2026-02-04
**Status**: Figure 1 fix ready to apply
**Next**: Copy FIXED version and recompile
