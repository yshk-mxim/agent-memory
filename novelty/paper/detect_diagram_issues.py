#!/usr/bin/env python3
"""
Automated TikZ Diagram Quality Checker
Detects text overlaps, spacing issues, and LaTeX warnings
"""

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json

def detect_text_overlap(pdf_path: str, page_number: int, tolerance: float = 2.0) -> List[Dict]:
    """
    Detect overlapping text bounding boxes in PDF

    Args:
        pdf_path: Path to PDF file
        page_number: Page number (0-indexed)
        tolerance: Minimum gap (in points) to not consider overlap

    Returns:
        List of overlap dictionaries
    """
    overlaps = []

    with pdfplumber.open(pdf_path) as pdf:
        if page_number >= len(pdf.pages):
            print(f"Error: Page {page_number} does not exist (only {len(pdf.pages)} pages)")
            return overlaps

        page = pdf.pages[page_number]
        words = page.extract_words()

        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                if boxes_overlap(word1, word2, tolerance):
                    overlap_area = calculate_overlap_area(word1, word2)
                    overlaps.append({
                        'text1': word1['text'],
                        'text2': word2['text'],
                        'bbox1': (round(word1['x0'], 2), round(word1['top'], 2),
                                 round(word1['x1'], 2), round(word1['bottom'], 2)),
                        'bbox2': (round(word2['x0'], 2), round(word2['top'], 2),
                                 round(word2['x1'], 2), round(word2['bottom'], 2)),
                        'overlap_area': round(overlap_area, 2),
                        'page': page_number + 1  # Human-readable page number
                    })

    return overlaps

def boxes_overlap(box1: Dict, box2: Dict, tolerance: float = 2.0) -> bool:
    """Check if two bounding boxes overlap (with tolerance for spacing)"""
    return not (box1['x1'] + tolerance < box2['x0'] or  # box1 left of box2
                box1['x0'] > box2['x1'] + tolerance or  # box1 right of box2
                box1['bottom'] + tolerance < box2['top'] or  # box1 above box2
                box1['top'] > box2['bottom'] + tolerance)  # box1 below box2

def calculate_overlap_area(box1: Dict, box2: Dict) -> float:
    """Calculate overlapping area between two boxes"""
    x_overlap = max(0, min(box1['x1'], box2['x1']) - max(box1['x0'], box2['x0']))
    y_overlap = max(0, min(box1['bottom'], box2['bottom']) - max(box1['top'], box2['top']))
    return x_overlap * y_overlap

def analyze_latex_warnings(log_file: str) -> Dict[str, List]:
    """
    Extract TikZ and layout warnings from LaTeX .log file
    """
    warnings = {
        'overfull_hbox': [],
        'underfull_hbox': [],
        'tikz_errors': [],
        'float_placement': [],
        'undefined_refs': [],
        'missing_chars': []
    }

    if not Path(log_file).exists():
        print(f"Warning: Log file not found: {log_file}")
        return warnings

    with open(log_file, 'r', encoding='latin-1', errors='ignore') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Overfull boxes (text extending past margins)
        if 'Overfull \\hbox' in line:
            match = re.search(r'(\d+\.?\d*)pt too wide', line)
            amount = match.group(1) if match else 'unknown'
            context = lines[i+1].strip() if i+1 < len(lines) else ''
            warnings['overfull_hbox'].append({
                'line_num': i + 1,
                'amount': amount + 'pt',
                'context': context[:100]
            })

        # Underfull boxes (poorly spaced text)
        elif 'Underfull \\hbox' in line:
            match = re.search(r'badness (\d+)', line)
            badness = match.group(1) if match else 'unknown'
            warnings['underfull_hbox'].append({
                'line_num': i + 1,
                'badness': badness
            })

        # TikZ-specific errors
        elif 'LaTeX Error' in line or '! ' in line:
            context = ''.join(lines[max(0,i-2):min(len(lines),i+5)])
            if 'tikz' in context.lower() or 'pgf' in context.lower():
                warnings['tikz_errors'].append({
                    'line_num': i + 1,
                    'message': line.strip(),
                    'context': context[:300]
                })

        # Float placement issues
        elif 'float specifier changed' in line.lower():
            warnings['float_placement'].append({
                'line_num': i + 1,
                'message': line.strip()
            })

        # Undefined references
        elif 'undefined' in line.lower() and ('reference' in line.lower() or 'citation' in line.lower()):
            warnings['undefined_refs'].append({
                'line_num': i + 1,
                'message': line.strip()
            })

    return warnings

def analyze_tikz_file(tikz_file: str) -> Dict[str, any]:
    """
    Analyze TikZ source code for potential issues
    """
    issues = {
        'absolute_positioning': [],
        'magic_numbers': [],
        'unnamed_nodes': [],
        'inconsistent_units': []
    }

    if not Path(tikz_file).exists():
        print(f"Warning: TikZ file not found: {tikz_file}")
        return issues

    with open(tikz_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Detect absolute positioning (less flexible)
        if re.search(r'\\node.*at\s*\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)', line):
            if 'below=' not in line and 'above=' not in line and 'right=' not in line:
                issues['absolute_positioning'].append({
                    'line': i + 1,
                    'content': line.strip()
                })

        # Detect unnamed nodes (harder to reference)
        if '\\node[' in line and '] at' in line:
            if not re.search(r'\]\s*\([a-zA-Z_][a-zA-Z0-9_]*\)', line):
                issues['unnamed_nodes'].append({
                    'line': i + 1,
                    'content': line.strip()[:80]
                })

        # Detect mixed units (cm, pt, em in same file)
        units_in_line = re.findall(r'\d+\.?\d*(cm|pt|em|ex|in|mm)', line)
        if len(set(units_in_line)) > 1:
            issues['inconsistent_units'].append({
                'line': i + 1,
                'units': list(set(units_in_line)),
                'content': line.strip()[:80]
            })

    return issues

def generate_report(pdf_path: str, log_file: str, tikz_files: List[str]) -> str:
    """
    Generate comprehensive quality report
    """
    report = []
    report.append("="*70)
    report.append("DIAGRAM QUALITY AUDIT REPORT")
    report.append("="*70)
    report.append("")

    # Check text overlaps in key pages
    critical_pages = {
        2: "Figure 1 & 2 (Memory Architecture + System Design)",
        5: "Figure 3 (TTFT Scaling)",
        7: "Figure 4 (Staggered Arrivals)"
    }

    total_overlaps = 0
    for page_num, description in critical_pages.items():
        report.append(f"\n### Page {page_num + 1}: {description}")
        overlaps = detect_text_overlap(pdf_path, page_num)

        if overlaps:
            report.append(f"  ⚠️  {len(overlaps)} TEXT OVERLAPS DETECTED:")
            for overlap in overlaps:
                report.append(f"     - '{overlap['text1']}' overlaps '{overlap['text2']}'")
                report.append(f"       Box1: {overlap['bbox1']}")
                report.append(f"       Box2: {overlap['bbox2']}")
                report.append(f"       Overlap area: {overlap['overlap_area']} sq pts")
            total_overlaps += len(overlaps)
        else:
            report.append("  ✓ No text overlaps detected")

    # Check LaTeX warnings
    report.append("\n" + "="*70)
    report.append("LATEX COMPILATION WARNINGS")
    report.append("="*70)

    warnings = analyze_latex_warnings(log_file)

    if warnings['overfull_hbox']:
        report.append(f"\n⚠️  {len(warnings['overfull_hbox'])} Overfull \\hbox warnings:")
        for i, warn in enumerate(warnings['overfull_hbox'][:5]):  # Show first 5
            report.append(f"  {i+1}. {warn['amount']} too wide - {warn['context'][:60]}")
        if len(warnings['overfull_hbox']) > 5:
            report.append(f"  ... and {len(warnings['overfull_hbox']) - 5} more")

    if warnings['tikz_errors']:
        report.append(f"\n🔴 {len(warnings['tikz_errors'])} TikZ errors found:")
        for err in warnings['tikz_errors']:
            report.append(f"  - Line {err['line_num']}: {err['message']}")

    # Analyze TikZ source files
    report.append("\n" + "="*70)
    report.append("TIKZ SOURCE CODE ANALYSIS")
    report.append("="*70)

    for tikz_file in tikz_files:
        report.append(f"\n### {Path(tikz_file).name}")
        issues = analyze_tikz_file(tikz_file)

        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            report.append("  ✓ No code quality issues detected")
        else:
            if issues['absolute_positioning']:
                report.append(f"  ⚠️  {len(issues['absolute_positioning'])} absolute positioning instances (prefer relative)")
            if issues['unnamed_nodes']:
                report.append(f"  ⚠️  {len(issues['unnamed_nodes'])} unnamed nodes (harder to reference)")
            if issues['inconsistent_units']:
                report.append(f"  ⚠️  {len(issues['inconsistent_units'])} lines with mixed units")

    # Summary
    report.append("\n" + "="*70)
    report.append("SUMMARY")
    report.append("="*70)
    report.append(f"Total text overlaps: {total_overlaps}")
    report.append(f"Overfull hbox warnings: {len(warnings['overfull_hbox'])}")
    report.append(f"TikZ errors: {len(warnings['tikz_errors'])}")

    if total_overlaps > 0:
        report.append("\n🔴 CRITICAL: Text overlaps detected. Figures need immediate fixes.")
    elif warnings['overfull_hbox']:
        report.append("\n⚠️  WARNING: Layout issues detected. Review recommended.")
    else:
        report.append("\n✅ PASS: No critical visual issues detected.")

    return "\n".join(report)

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze PDF and LaTeX files for diagram quality issues')
    parser.add_argument('--pdf', default='semantic_colm2026.pdf', help='PDF file to analyze')
    parser.add_argument('--log', default='semantic_colm2026.log', help='LaTeX log file')
    parser.add_argument('--tikz-dir', default='figures/', help='Directory containing TikZ files')
    parser.add_argument('--output', default='DIAGRAM_QUALITY_REPORT.txt', help='Output report file')

    args = parser.parse_args()

    # Find all TikZ files
    tikz_files = list(Path(args.tikz_dir).glob('fig_*.tex'))

    # Generate report
    report = generate_report(
        pdf_path=args.pdf,
        log_file=args.log,
        tikz_files=[str(f) for f in tikz_files]
    )

    # Print to console
    print(report)

    # Save to file
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {args.output}")

if __name__ == '__main__':
    main()
