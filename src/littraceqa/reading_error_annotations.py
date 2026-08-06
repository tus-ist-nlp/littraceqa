"""Validation-only annotations used by post-inference error analysis.

These values are isolated from the generic analyzer and are never imported by
the inference runner, so validation knowledge cannot enter an AOAI prompt.
"""

KNOWN_DATASET_ISSUES: dict[str, tuple[str, ...]] = {
    "q_001": (
        "The registered gold evidence contains only 15.66, while deriving the "
        + "gold answer 14.70 also requires the ICAE value.",
    ),
    "q_054": (
        "The question asks for AP^kit_3D, while the gold table schema and "
        + "evidence target AP^nus_3D.",
    ),
    "q_056": (
        "The question says AP-BPTT, while the gold answer row says AT-BPTT.",
    ),
}

# q_031--q_051 are multiple-choice comparisons whose gold annotations may also
# contain evidence for distractors. Manual review established the papers needed
# to derive the correct meaning-level answer.
ANSWER_REQUIRED_PAPER_IDS: dict[str, tuple[str, ...]] = {
    "q_031": ("iclr2025_03463",),
    "q_032": ("iclr2025_03463",),
    "q_033": ("iclr2025_00615",),
    "q_034": ("icml2025_01371",),
    "q_035": ("iclr2025_03031",),
    "q_036": ("iclr2025_03463",),
    "q_037": ("iclr2025_00706",),
    "q_038": ("iclr2025_00911",),
    "q_039": ("icml2025_01371",),
    "q_040": ("icml2025_01371",),
    "q_041": ("iclr2025_03031",),
    "q_042": ("iclr2025_03031", "icml2025_01371"),
    "q_043": ("cvpr2025_00533", "cvpr2025_01683"),
    "q_044": ("iclr2025_00978", "icml2025_00188"),
    "q_045": ("iclr2025_02715", "acl2025_01863"),
    "q_046": ("cvpr2025_00533",),
    "q_047": ("iclr2025_00978",),
    "q_048": ("acl2025_01863",),
    "q_049": ("neurips2025_05262",),
    "q_050": ("icml2025_00188",),
    "q_051": ("neurips2025_03461",),
}
