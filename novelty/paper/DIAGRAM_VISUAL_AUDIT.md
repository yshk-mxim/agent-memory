# Visual Audit: Diagrams and Charts
**Date**: 2026-02-04
**PDF**: semantic_colm2026.pdf
**Status**: CRITICAL ISSUES FOUND

---

## 🔴 CRITICAL ISSUES IDENTIFIED

### Figure 1 (Page 3): Memory Architecture Comparison - **MANGLED**

**Problems:**
1. **Text overlap**: In the left diagram (Discrete GPU), bandwidth annotations overlap:
   - "A100: 1,555 GB/s (VRAM)"
   - "PCIe 4.0: 32 GB/s"
   - These appear to be on top of each other

2. **Awkward line breaking**: "PCIe transfer" text has poor positioning

3. **Spacing issues**: "Explicit copy required" text is cramped near the arrow

4. **Bottom annotation problems**: The blue text "Bandwidth convergence: edge UMA = entry datacenter UMA" has layout issues

**Visual Quality**: 3/10 - Functional but unprofessional, text readability severely impacted

---

### Figure 2 (Page 3): System Architecture - **MODERATE ISSUES**

**Problems:**
1. **Disproportionate elements**: The "Disk" circle on the right is too large compared to other elements

2. **Annotation positioning**: Bottom text ("2.0-4.3× E2E speedup", "72% memory savings", "81.6× TTFT (hot)") appears cramped and poorly aligned

3. **Uneven spacing**: Agent boxes (Agent 1, 2, 3) have inconsistent vertical spacing

4. **Flow clarity**: Arrows between components could be more prominent

**Visual Quality**: 6/10 - Functional but needs polish

---

## ✅ WELL-EXECUTED FIGURES

### Figure 3 (Page 6): TTFT Scaling Chart - **EXCELLENT**

**Strengths:**
- Clean pgfplots line chart with proper log-log scaling
- Legend is clear and well-positioned
- Annotations (10.5×, 81.6×) are optimally placed
- Three distinct line styles (Cold/Warm/Hot) easily distinguishable
- Grid lines enhance readability
- Professional publication quality

**Visual Quality**: 9/10 - Publication ready

---

### Figure 4 (Page 8): Staggered Arrivals Chart - **GOOD**

**Strengths:**
- Clean grouped bar chart
- Clear color coding (purple/blue vs green)
- Annotations readable and well-placed
- Proper use of whitespace

**Minor improvements needed:**
- Text annotations ("4% penalty", "2.6× faster") could be 1-2pt larger
- Bar width could be slightly increased for emphasis

**Visual Quality**: 8/10 - Nearly publication ready

---

## 📋 TABLES - ALL CLEAN

**Table 1-4**: All tables use booktabs formatting correctly, no issues detected.

---

## 🛠️ AUTOMATED FIX METHODS

### Method 1: Claude/GPT-4 with Vision for TikZ Debugging

**Tools:**
- Claude 3.5 Sonnet (vision + code)
- GPT-4o (vision + code generation)

**Workflow:**
1. **Input**: Screenshot of mangled figure + TikZ source code
2. **Prompt**: "This TikZ diagram has overlapping text and spacing issues. Analyze the visual output and the LaTeX source, then provide corrected TikZ code with proper node positioning, spacing, and text placement."
3. **Output**: Corrected TikZ code with specific node coordinates adjusted

**Example prompt for Figure 1:**
```
I have a TikZ diagram comparing UMA vs Discrete GPU memory architectures.
Visual issues:
- Text annotations overlap at "A100: 1,555 GB/s" and "PCIe 4.0: 32 GB/s"
- "Explicit copy required" text is cramped
- Bottom bandwidth comparison text has layout problems

Please analyze the attached screenshot and TikZ code, then provide:
1. Specific coordinate adjustments to eliminate overlaps
2. Better text positioning using proper anchor points
3. Improved spacing between elements
4. Corrected code with comments explaining changes
```

---

### Method 2: TikZ Linting and Validation Tools

**Tools:**

1. **lacheck** (LaTeX syntax checker)
```bash
lacheck figures/fig_uma_comparison.tex
```
- Detects: syntax errors, undefined references
- Does NOT detect: visual layout issues

2. **chktex** (Advanced LaTeX checker)
```bash
chktex figures/fig_uma_comparison.tex
```
- Detects: stylistic issues, potential problems
- Does NOT detect: visual overlaps

3. **TikZiT** (Interactive TikZ editor)
- Visual WYSIWYG editor for TikZ diagrams
- Allows drag-and-drop node repositioning
- Export clean TikZ code
- Download: https://tikzit.github.io/

4. **tikzedt** (TikZ editor with live preview)
- Real-time compilation and preview
- Syntax highlighting
- Coordinate inspector
- GitHub: https://github.com/hchapman/tikzedt (older, less maintained)

---

### Method 3: Python-Based TikZ Optimization

**Tool: `tikzplotlib`** (for pgfplots charts only)
```python
import matplotlib.pyplot as plt
import tikzplotlib

# Create clean matplotlib plot
fig, ax = plt.subplots()
# ... plot data ...

# Export to clean TikZ
tikzplotlib.save("clean_figure.tex")
```

**Limitation**: Only works for plots/charts, not for block diagrams like Figure 1 and 2.

---

### Method 4: Overleaf Visual Debugger

**Process:**
1. Upload TikZ files to Overleaf
2. Use built-in PDF preview with source-to-PDF sync
3. Click on visual element → jumps to source code
4. Adjust coordinates iteratively with immediate visual feedback
5. Overleaf highlights compilation errors in real-time

**Advantage**: No local setup, collaborative debugging

---

## 🔧 SPECIFIC FIXES FOR IDENTIFIED ISSUES

### Fix for Figure 1 (fig_uma_comparison.tex)

**Problem areas in current code:**

```latex
% Current (BROKEN):
\node[below=1.5cm of discrete_title, font=\small, align=center] {
    A100: 1,555 GB/s (VRAM)\\
    PCIe 4.0: ~32 GB/s
};

\node[below=1.5cm of uma_title, font=\small, align=center] {
    M4 Pro: 273 GB/s\\
    DGX Spark: 273 GB/s
};
```

**Issues:**
1. Both use `below=1.5cm` which positions them at same Y-coordinate relative to different anchors
2. This causes overlap when titles are at different heights
3. Need absolute positioning or better relative positioning

**Proposed fix:**

```latex
% FIXED VERSION:
% Left side bandwidth annotation
\node[below=2.0cm of gpu_compute, font=\small, align=center] (left_bandwidth) {
    A100 VRAM: 1,555 GB/s\\
    PCIe 4.0 transfer: 32 GB/s
};

% Right side bandwidth annotation
\node[below=2.0cm of cpu_gpu, font=\small, align=center] (right_bandwidth) {
    M4 Pro: 273 GB/s\\
    DGX Spark: 273 GB/s
};

% Convergence note (properly positioned BELOW bandwidth notes)
\node[below=0.5cm of right_bandwidth, font=\footnotesize, text=blue!70!black, align=center] {
    Bandwidth convergence:\\
    edge UMA = entry datacenter UMA
};
```

**Key changes:**
1. Changed anchor from title to bottom components (gpu_compute, cpu_gpu)
2. Increased spacing from 1.5cm to 2.0cm
3. Named nodes for better positioning control
4. Clarified text labels to avoid ambiguity
5. Positioned convergence note relative to bandwidth annotation, not title

---

### Fix for Figure 2 (fig_architecture.tex)

**Problem:** Disk circle too large, bottom annotations cramped

**Current (BROKEN):**
```latex
% Assuming current code has large disk circle
\node[circle, draw, minimum size=3cm] (disk) {...};

% Bottom annotations likely positioned absolutely
\node at (someX, someY) {2.0-4.3× E2E speedup};
\node at (someX2, someY2) {72% memory savings};
```

**Proposed fix:**

```latex
% FIXED: Proportional disk node
\node[circle, draw, fill=yellow!30, minimum size=2cm, font=\small] (disk) at (10,1) {
    Disk\\
    safetensors\\
    50ms reload
};

% FIXED: Properly spaced annotations below diagram
\node[below=0.3cm of block_pool, font=\small, color=blue!70!black] {
    2.0--4.3$\times$ E2E speedup
};

\node[below=0.3cm of q4_pipeline, font=\small, color=green!70!black] {
    72\% memory savings
};

\node[below=0.3cm of disk, font=\small, color=orange!70!black] {
    81.6$\times$ TTFT (hot)
};
```

**Key changes:**
1. Reduced disk circle from 3cm to 2cm
2. Positioned annotations relative to components above them (not absolute)
3. Used consistent spacing (0.3cm below)
4. Added color coding for visual distinction

---

## 📊 AUTOMATED DETECTION METHODS

### Method 1: PDF Visual Analysis with Python

**Tool: `pdfplumber` + `Pillow` + OpenCV**

```python
import pdfplumber
from PIL import Image
import numpy as np
import cv2

def detect_text_overlap(pdf_path, page_number):
    """
    Detect overlapping text bounding boxes in PDF
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]

        # Extract all text with bounding boxes
        words = page.extract_words()

        overlaps = []
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                # Check if bounding boxes overlap
                if boxes_overlap(word1, word2):
                    overlaps.append({
                        'text1': word1['text'],
                        'text2': word2['text'],
                        'bbox1': (word1['x0'], word1['top'], word1['x1'], word1['bottom']),
                        'bbox2': (word2['x0'], word2['top'], word2['x1'], word2['bottom'])
                    })

        return overlaps

def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap"""
    return not (box1['x1'] < box2['x0'] or  # box1 left of box2
                box1['x0'] > box2['x1'] or  # box1 right of box2
                box1['bottom'] < box2['top'] or  # box1 above box2
                box1['top'] > box2['bottom'])  # box1 below box2

# Usage
overlaps = detect_text_overlap('semantic_colm2026.pdf', page_number=2)  # Page 3 (0-indexed as 2)
for overlap in overlaps:
    print(f"OVERLAP DETECTED: '{overlap['text1']}' and '{overlap['text2']}'")
```

---

### Method 2: LaTeX Compilation Warnings Analysis

**Tool: Parse LaTeX .log file for warnings**

```python
def analyze_latex_warnings(log_file):
    """
    Extract TikZ and overfull/underfull warnings from LaTeX log
    """
    warnings = {
        'overfull_hbox': [],
        'underfull_hbox': [],
        'tikz_errors': [],
        'float_placement': []
    }

    with open(log_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'Overfull \\hbox' in line:
            warnings['overfull_hbox'].append({
                'line': i,
                'message': line.strip(),
                'context': lines[i+1] if i+1 < len(lines) else ''
            })

        elif 'Underfull \\hbox' in line:
            warnings['underfull_hbox'].append({
                'line': i,
                'message': line.strip()
            })

        elif 'LaTeX Error' in line and 'TikZ' in ''.join(lines[max(0,i-5):i+5]):
            warnings['tikz_errors'].append({
                'line': i,
                'message': line.strip(),
                'context': ''.join(lines[max(0,i-2):min(len(lines),i+3)])
            })

        elif 'float specifier changed' in line:
            warnings['float_placement'].append({
                'line': i,
                'message': line.strip()
            })

    return warnings

# Usage
warnings = analyze_latex_warnings('semantic_colm2026.log')
print(f"Overfull hboxes: {len(warnings['overfull_hbox'])}")
print(f"TikZ errors: {len(warnings['tikz_errors'])}")
```

---

### Method 3: Claude API for Batch TikZ Review

**Automated workflow:**

```python
import anthropic
from pathlib import Path

def review_tikz_figure(tikz_code, pdf_screenshot_path):
    """
    Use Claude API to analyze TikZ code and visual output
    """
    client = anthropic.Anthropic(api_key="your-api-key")

    with open(pdf_screenshot_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

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
                    "text": f"""Analyze this TikZ diagram for visual quality issues:

TikZ Source Code:
```latex
{tikz_code}
```

Identify:
1. Text overlaps or cramped spacing
2. Disproportionate element sizes
3. Poor alignment or positioning
4. Unclear visual hierarchy

Provide:
1. Specific issues found (with coordinates)
2. Corrected TikZ code
3. Explanation of changes

Format as:
## Issues
1. ...
2. ...

## Corrected Code
```latex
...
```

## Explanation
..."""
                }
            ]
        }]
    )

    return message.content[0].text

# Batch process all figures
figure_files = Path('figures/').glob('fig_*.tex')
for fig_file in figure_files:
    tikz_code = fig_file.read_text()
    # Screenshot would need to be extracted from compiled PDF
    review = review_tikz_figure(tikz_code, f"screenshots/{fig_file.stem}.png")
    print(f"\n{'='*60}\n{fig_file.name}\n{'='*60}\n{review}")
```

---

## 🎯 RECOMMENDED IMMEDIATE ACTIONS

### Priority 1: Fix Figure 1 (Critical)

**Steps:**
1. Use TikZ coordinate debugging approach
2. Apply the specific fixes documented above
3. Recompile and visually verify
4. Use pdfplumber overlap detection to confirm no text overlaps remain

**Time estimate**: 30 minutes

---

### Priority 2: Polish Figure 2 (Moderate)

**Steps:**
1. Reduce disk circle size to 2cm
2. Reposition bottom annotations using relative positioning
3. Adjust agent box spacing to be uniform
4. Recompile and verify

**Time estimate**: 20 minutes

---

### Priority 3: Minor tweaks to Figure 4 (Low)

**Steps:**
1. Increase annotation font size from \small to \normalsize
2. Recompile and verify readability

**Time estimate**: 5 minutes

---

## 📚 BEST PRACTICES FOR FUTURE TikZ DIAGRAMS

### 1. Use Named Nodes
```latex
% GOOD
\node[component] (block_pool) at (3,2) {Block Pool};
\node[below=1cm of block_pool] (annotation) {Details};

% BAD
\node[component] at (3,2) {Block Pool};
\node at (3,0.5) {Details};  % Absolute positioning breaks if diagram changes
```

### 2. Relative Positioning Over Absolute
```latex
% GOOD
\node[below=0.5cm of previous_node] {Text};

% BAD
\node at (5.3, 2.7) {Text};  % Magic numbers
```

### 3. Test at Multiple Scales
```latex
% Add scale testing in preamble
\tikzset{scale=0.8}  % Test if diagram works at 80%
\tikzset{scale=1.2}  % Test if diagram works at 120%
```

### 4. Use Consistent Units
```latex
% Stick to one unit system throughout
node distance=1cm  % NOT mixing cm, pt, em
minimum width=2cm
```

### 5. Automated Bounding Box Checks
```latex
% Add to TikZ code for debugging
\draw[red, dashed] (current bounding box.south west) rectangle (current bounding box.north east);
```

---

## 🔍 QUALITY CHECKLIST FOR EACH FIGURE

Before finalizing any TikZ diagram:

- [ ] No text overlaps (verify with pdfplumber script)
- [ ] All text readable at 100% zoom
- [ ] Consistent spacing between elements (use named distances)
- [ ] Proper visual hierarchy (important elements more prominent)
- [ ] Accessible colors (test in grayscale)
- [ ] No "Overfull hbox" warnings for the figure
- [ ] Figure renders correctly when scaled to 80% and 120%
- [ ] All arrows point to correct targets
- [ ] Caption accurately describes visual content
- [ ] Figure is referenced in main text before it appears

---

## 📊 SUMMARY SCORES

| Figure | Quality | Critical Issues | Action Needed |
|--------|---------|-----------------|---------------|
| Figure 1 (UMA) | 3/10 | Text overlap, cramped layout | **URGENT FIX** |
| Figure 2 (System) | 6/10 | Spacing, proportions | **POLISH** |
| Figure 3 (TTFT) | 9/10 | None | None - excellent |
| Figure 4 (Staggered) | 8/10 | Minor text size | Optional tweak |
| Algorithm 1 | 9/10 | None | None |
| Tables 1-4 | 10/10 | None | None |

**Overall Paper Diagram Quality**: 7.2/10 (dragged down by Figure 1)

**With fixes**: Would improve to 8.5/10 (publication quality)

---

## 🚀 AUTOMATED TOOLCHAIN RECOMMENDATION

**Best setup for TikZ quality assurance:**

1. **TikZiT** - Interactive editing and visual debugging
2. **Overleaf** - Real-time compilation and preview
3. **pdfplumber** - Automated text overlap detection
4. **Claude API** - AI-powered code review and fixes
5. **lacheck + chktex** - Syntax validation

**Workflow:**
```
1. Create diagram in TikZiT (visual)
   ↓
2. Export to LaTeX, upload to Overleaf
   ↓
3. Fine-tune coordinates with live preview
   ↓
4. Run pdfplumber overlap detection
   ↓
5. If issues found, use Claude API for fix suggestions
   ↓
6. Final syntax check with lacheck
   ↓
7. Compile final PDF
```

---

**Status**: AUDIT COMPLETE
**Next**: Implement Priority 1 fix for Figure 1
