"""Prompt fragments shared across the discrete, science, and CARS generators.

Single source of truth so a rule hardened into one pipeline stays in sync across
all of them. Two prior fixes (the Phase 2.1 "no fifth option" hardening and the
Phase 1.1 LaTeX-notation rule) each started life in ONE prompt and had to be
hand-copied to the others, which is exactly how the discrete/science prompts
drifted out of sync. Centralizing the wording here prevents that drift.

The CARS prompt keeps its own (already-working) inline wording for the
no-fifth-option rule to avoid any risk of regressing its rendered text; the
NO_FIFTH_OPTION_* constants reproduce that rule's MEANING for the discrete and
science question prompts.
"""

# --- No-fifth-option contract (EXACTLY four options A-D, never an option E) ---
#
# Mirrors the wording hardened into the CARS per-question prompt in Phase 2.1.
# Three pieces, dropped into the matching section of a generation prompt:
#   * NO_FIFTH_OPTION_RULE             -> general rules (the choice set itself)
#   * NO_FIFTH_OPTION_EXPLANATION_RULE -> the explanation rule
#   * NO_FIFTH_OPTION_CONTRACT         -> the strict output contract

NO_FIFTH_OPTION_RULE = (
    "There are EXACTLY four options: A, B, C, and D. There is NO option E (or F, "
    "or any option beyond D). The \"choices\" object must contain EXACTLY the four "
    "keys \"A\", \"B\", \"C\", and \"D\" and nothing else — no extra keys (no \"E\", "
    "no \"C2\", no duplicates), no missing/empty keys; each choice a non-empty "
    "string. Exactly ONE answer is unambiguously correct."
)

NO_FIFTH_OPTION_EXPLANATION_RULE = (
    "The explanation must reference ONLY options A, B, C, and D; NEVER mention, "
    "discuss, or reference an option E or any option beyond D — no such option "
    "exists. Do NOT write phrases like \"Option E\", \"E is...\", or \"E "
    "overstates...\". If you find yourself about to reference a fifth option, "
    "stop: there are only four."
)

NO_FIFTH_OPTION_CONTRACT = (
    "\"choices\" must contain EXACTLY the keys A, B, C, D and nothing else — no "
    "\"E\" or any further option. The \"explanation\" must not reference any "
    "option key outside {A, B, C, D}; there is no option E, so never write "
    "\"Option E\" or reason about a fifth choice."
)


# --- Choice-reference bolding in explanations (shared by all generators) ------
#
# When an explanation refers to an answer choice, the writer names it "Choice X"
# and bolds that whole phrase in markdown. Doing this GENERATION-side — where the
# model knows what is a choice versus a chemical element symbol or experimental
# label — replaces a brittle render-time regex in the consuming app that mis-bolded
# bare letters (e.g. the C in a "C–Br" bond, the A in "Series A"). Bolding the
# PHRASE "Choice X", never a bare letter, keeps the reference unambiguous.
CHOICE_REFERENCE_BOLD_RULE = (
    "When the explanation refers to an answer choice, name it \"Choice A/B/C/D\" "
    "and bold that whole phrase in markdown — e.g. \"**Choice B** incorrectly "
    "attributes the rise...\", \"**Choice A** describes a competitive inhibitor\", "
    "\"**Choice D** is correct because...\". Bold the entire \"Choice X\" phrase, "
    "never a bare letter, and do NOT bold letters that are chemical element symbols "
    "(the C in a C–Br bond), experimental labels (Series A, Group B), or variables "
    "— only explicit \"Choice X\" references."
)


# --- LaTeX math/chemical notation contract (shared by all generators) --------
#
# All NEW questions replace the old bank, so math/chemistry must render. Every
# mathematical or chemical token in any output field is written as LaTeX so a
# downstream renderer can typeset it; bare underscores/carets/asterisks are
# banned for notation. Ordinary prose is left alone — only the notation itself is
# wrapped. (Originally defined in prompts.science_passage during Phase 1.1; moved
# here verbatim so the discrete and science question prompts share one source of
# truth. The rendered text is unchanged.)
LATEX_NOTATION_RULE = (
    "MATH & CHEMISTRY NOTATION (LaTeX, REQUIRED): Write ALL mathematical and chemical "
    "notation as LaTeX, delimited with $...$ for inline (use $$...$$ ONLY for a genuine "
    "display equation). This applies to every field where such notation appears — passage "
    "prose, table cells, question stems, answer choices, and explanations. Examples: "
    r"subscripts (K_M -> $K_\text{M}$), exponents (10^-4 -> $10^{-4}$), Greek letters "
    r"($\rho$, $\mu$, $\Delta$), units and products ($\rho V g$, $5.0\ \text{V/cm}$, "
    r"$3.0\ \text{mol/L}$), and chemical formulas/ions (H2O -> $\text{H}_2\text{O}$, "
    r"HCO3- -> $\text{HCO}_3^-$, Na+ -> $\text{Na}^+$). For the micro symbol / micro-units use "
    r"\mu (e.g. $\mu\text{m/s}$, $\mu\text{M}$, $\mu\text{g}$), and NEVER \textmu or \textmicro "
    r"(KaTeX supports neither). NEVER use bare underscores, carets, "
    "or asterisks for math or chemistry in any field. Do NOT LaTeX-wrap ordinary prose — "
    "wrap ONLY the mathematical/chemical notation itself. "
    "JSON ESCAPING: your output is a JSON object, so every LaTeX backslash inside a string "
    r"value MUST be escaped as a DOUBLE backslash \\ to keep the JSON valid — write "
    r'"$K_\\text{a}$", "$\\times$", "$\\rho$", "$\\Delta H$" (each command keeps its single '
    "backslash once JSON-decoded). Keep each question reasonably concise so the JSON is "
    "complete and well-formed."
)

# One-line note for the review/solve models so LaTeX-delimited notation in the
# content they read is understood and NOT flagged as a formatting error.
LATEX_REVIEW_NOTE = (
    "Mathematical and chemical notation is written in LaTeX (delimited with $...$ or "
    "$$...$$); read it as ordinary math/chemistry and do NOT flag the LaTeX itself as an "
    "error."
)
