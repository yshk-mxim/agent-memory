# COLM 2026 Quick Reference Card

## LaTeX Template Structure
```
papers_archive/Template-master/
├── colm2026_conference.tex      ← Main document template
├── colm2026_conference.sty      ← Style file (MANDATORY)
├── colm2026_conference.bst      ← Bibliography style (natbib)
├── colm2026_conference.bib      ← Example bibliography
├── colm2026_conference.pdf      ← Compiled example
├── math_commands.tex            ← Math notation definitions
└── README.md
```

## Essential COLM Formatting

| Element | Size | Font | Notes |
|---------|------|------|-------|
| Title | 17pt | Palatino Bold | Left-aligned |
| Section | 14pt | Palatino Bold | Numbered (automatic) |
| Subsection | 12pt | Palatino Bold | Numbered (automatic) |
| Body | 10pt | Palatino | 11pt line spacing |
| Figure/Table Font | 10pt | Palatino | Same as body |
| Margins | 1.5in left | — | 5.5in text width, 9in height |

## Document Options

```latex
\usepackage[submission]{colm2026_conference}  % During review (anonymous)
\usepackage[final]{colm2026_conference}       % After acceptance (with authors)
\usepackage[preprint]{colm2026_conference}    % Preprint version
```

## Key Packages (already included in colm2026_conference.sty)

```latex
\usepackage{tgpagella}          % Palatino font (text)
\usepackage{mathpazo}           % Palatino font (math)
\usepackage{inconsolata}        % Monospace for code
\usepackage{natbib}             % Author-year citations
\usepackage{fancyhdr}           % Headers/footers
\usepackage{microtype}          % Better spacing
\usepackage{hyperref}           % Hyperlinks (dark blue)
\usepackage{booktabs}           % Table styling (no vertical lines)
```

## Compilation Chain

```bash
# Initial compilation
pdflatex colm2026_conference.tex

# Generate bibliography
bibtex colm2026_conference

# Update references
pdflatex colm2026_conference.tex
pdflatex colm2026_conference.tex  # Run twice to resolve all cross-refs
```

## Citation Examples (natbib style)

```latex
\citet{Liu2024}                    % Liu (2024)
\citep{Liu2024}                    % (Liu, 2024)
\citep{Liu2024,Kim2025}            % (Liu, 2024; Kim, 2025)
\citep{Liu2024, p.\ 42}            % (Liu, 2024, p. 42)

In narrative:
Liu et al.~\citep{Liu2024} demonstrated...
```

## Figure Template

```latex
\begin{figure}[t]  % [t]=top, [b]=bottom, [h]=here
\centering
\includegraphics[width=5.5in]{figures/example.pdf}
% OR for ASCII/TikZ figures:
% \input{figures/example.tex}
\caption{Brief title. Explanation of what to see
  in the figure and key takeaway. Reference specific elements
  (left plot shows X, right plot shows Y).}
\label{fig:example}
\end{figure}

In text: As shown in Figure~\ref{fig:example}, ...
```

## Table Template

```latex
\begin{table}[t]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Column A} & \textbf{Column B} & \textbf{Column C} \\
\midrule
Row 1 & 100 & 2.5x \\
Row 2 & 200 & 3.1x \\
\bottomrule
\end{tabular}
\caption{What the table shows and why it matters.
  Key findings highlighted.}
\label{tab:example}
\end{table}

In text: Table~\ref{tab:example} shows that...
```

## Section Hierarchy

```latex
\section{Introduction}
\section{Background & Motivation}
  \subsection{Re-Prefill Problem}
  \subsection{Apple Silicon UMA}
\section{System Design}
  \subsection{Block Pool Architecture}
  \subsection{Q4 Persistence Pipeline}
\section{Evaluation}
\section{Discussion}
\section{Related Work}
\section{Conclusion}

\appendix
\section{Appendix A: Cache Format}
\section{Appendix B: Lazy Evaluation}

\bibliography{colm2026_conference}  % Will use colm2026_conference.bib
```

## Page Budget for Semantic Paper

| Section | Pages | Notes |
|---------|-------|-------|
| Title + Abstract | 0.5 | On first page with intro |
| Introduction | 1 | High-level problem + contributions |
| Background | 0.75 | UMA hardware context |
| System Design | 2.5 | Core technical contribution |
| Implementation | 0.75 | Concrete systems details |
| Evaluation | 1.5 | Key results only |
| Discussion | 1.5 | Novelty, positioning, limitations |
| Related Work | 1 | Condensed, 2-3 sentences per system |
| Conclusion | 0.5 | Recap + future work |
| **TOTAL** | **9.0** | **Hard limit** |
| References | ∞ | Unlimited (separate pages) |
| Appendices | ∞ | Optional |

## Line Length Check

```
If your line in the text editor is longer than ~75 characters,
it will likely exceed the 5.5-inch COLM column width in the PDF.
Break long sentences or reduce indentation nesting.
```

## Color Scheme (if using color)

```
Text: Black (#000000)
Links: Dark blue (#000080) — automatically applied by hyperref
Figures: Use restrained palette:
  - Blue (#4a9eff) for data/storage
  - Yellow (#ffd93d) for processing
  - Green (#2ecc71) for results
  - Red (#ff6b6b) for errors/warnings
  - Gray (#cccccc) for backgrounds only
```

## Common Mistakes to Avoid

❌ Changing \usepackage order (breaks COLM style)
❌ Using geometry package without [pass] flag
❌ Vertical lines in tables (violates booktabs)
❌ Numbered citations [1], [2] (use natbib author-year)
❌ Author names in submission mode (should be anonymous)
❌ Equations without \label{} and cross-references
❌ Figures without \caption{}
❌ Main text > 9 pages
❌ Non-Palatino fonts in body
❌ Inline-header lists with **bold** (use \section{} instead)

## File Organization for Submission

```
semantic_novelty.tex          ← Main file
semantic_novelty.bib          ← Bibliography
figures/
  ├── fig_ttft.pdf            ← High-resolution figures
  ├── fig_staggered.pdf
  └── ...
```

Save as single `.tex` file when submitting to OpenReview.

## Contact for COLM 2026

- Website: https://www.colmweb.org/
- Submission: https://openreview.net/
- Template Issues: COLM-org/Template on GitHub

---

**Ready to draft?** Start with the full `STYLE_GUIDE.md`. This quick ref is for checking specific rules during writing.
