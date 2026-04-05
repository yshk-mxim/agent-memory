#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch contextFMHARunner.cpp to support Thor sm_110.

Edge-LLM 0.6.0 has applyThorSMRenumberWAR in the attention plugin
but not in the context FMHA runner. This patch adds the same remap
(sm_110 -> sm_101) to the FMHA context path.
"""

import sys
from pathlib import Path


def patch(filepath: str) -> None:
    path = Path(filepath)
    src = path.read_text()

    # 1. Remap sm_110 -> sm_101 in getFMHAKernelList
    old = "FMHAKernelLoadHashKey hash_key{type, sm};"
    new = (
        "// Thor sm_110 -> sm_101 (matches applyThorSMRenumberWAR in plugin)\n"
        "        if (sm == 110) sm = 101;\n"
        "        FMHAKernelLoadHashKey hash_key{type, sm};"
    )
    if old in src and "sm == 110" not in src:
        src = src.replace(old, new)
        print("Patched: getFMHAKernelList sm remap")

    # 2. Remap in ContextFMHARunner constructor
    old2 = ", mSmVersion(smVersion)"
    new2 = ", mSmVersion(smVersion == 110 ? 101 : smVersion)"
    if old2 in src and "smVersion == 110" not in src:
        src = src.replace(old2, new2, 1)
        print("Patched: constructor sm remap")

    # 3. Add sm_110 to isSm10x
    old3 = "bool const isSm10x = (smVersion == fmha_v2::kSM_100 || smVersion == fmha_v2::kSM_101);"
    new3 = "bool const isSm10x = (smVersion == fmha_v2::kSM_100 || smVersion == fmha_v2::kSM_101 || smVersion == 110);"
    if old3 in src:
        src = src.replace(old3, new3)
        print("Patched: isSm10x check")

    path.write_text(src)
    print(f"Done: {filepath}")


if __name__ == "__main__":
    patch(sys.argv[1])
