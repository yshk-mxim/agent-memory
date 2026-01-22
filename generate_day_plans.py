#!/usr/bin/env python3
"""
Generate individual day-by-day plan files from complete_plan.md

This script parses complete_plan.md and extracts each day's section,
creating individual files day_01.md through day_21.md in the plans/ directory.
"""

import re
import os
from pathlib import Path


def extract_day_sections(complete_plan_path: str) -> dict:
    """
    Extract all day sections from the complete plan.

    Returns:
        dict: Mapping of day number to content
    """
    with open(complete_plan_path, 'r') as f:
        content = f.read()

    days = {}

    # Pattern to match day headers like "### Day 1 (Monday): Environment Setup"
    day_pattern = r'### Day (\d+) \((\w+)\): (.+?)\n\n(.*?)(?=\n### Day \d+|\n## WEEK |\Z)'

    matches = re.finditer(day_pattern, content, re.DOTALL)

    for match in matches:
        day_num = int(match.group(1))
        day_name = match.group(2)
        day_title = match.group(3)
        day_content = match.group(4)

        days[day_num] = {
            'day_name': day_name,
            'title': day_title,
            'content': day_content.strip()
        }

    return days


def create_day_file(day_num: int, day_data: dict, output_dir: str):
    """
    Create an individual day plan file.

    Args:
        day_num: Day number (1-21)
        day_data: Dictionary with day_name, title, content
        output_dir: Directory to save the file
    """
    # Determine week number
    week_num = ((day_num - 1) // 7) + 1

    # Create filename with zero-padding
    filename = f"day_{day_num:02d}.md"
    filepath = os.path.join(output_dir, filename)

    # Build the file content
    file_content = f"""# Day {day_num} ({day_data['day_name']}): {day_data['title']}

**Week {week_num} - Day {((day_num - 1) % 7) + 1}**

---

{day_data['content']}

---

## Quick Reference

**Previous Day:** [Day {day_num - 1}](day_{day_num - 1:02d}.md) (if exists)
**Next Day:** [Day {day_num + 1}](day_{day_num + 1:02d}.md) (if exists)
**Complete Plan:** [Complete 3-Week Plan](../complete_plan.md)

---

## Checklist for Today

- [ ] Review objectives and tasks
- [ ] Set up required files and dependencies
- [ ] Execute all tasks according to timeline
- [ ] Verify success criteria
- [ ] Document any issues or deviations
- [ ] Prepare for next day

---

*Generated from complete_plan.md*
"""

    with open(filepath, 'w') as f:
        f.write(file_content)

    print(f"✓ Created {filename}")


def main():
    """Main execution function"""
    # Paths
    base_dir = Path(__file__).parent
    complete_plan_path = base_dir / "complete_plan.md"
    output_dir = base_dir / "plans"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Generating Day-by-Day Plan Files")
    print("="*60)
    print(f"\nReading from: {complete_plan_path}")
    print(f"Output to: {output_dir}\n")

    # Extract days from complete plan
    days = extract_day_sections(str(complete_plan_path))

    print(f"Found {len(days)} days in complete plan\n")

    # Create individual day files
    for day_num in sorted(days.keys()):
        create_day_file(day_num, days[day_num], str(output_dir))

    print(f"\n{'='*60}")
    print(f"✓ Successfully generated {len(days)} day plan files!")
    print(f"{'='*60}")
    print(f"\nFiles created in: {output_dir}")
    print(f"\nTo view a specific day:")
    print(f"  cat plans/day_01.md")
    print(f"\nTo start the research sprint:")
    print(f"  cat plans/day_01.md | less")
    print()


if __name__ == "__main__":
    main()
