# mcat_topics.json Enrichment Audit — Phase 1 (Report Only)

**Source of truth:** `what's on the mcat pdf.pdf` (AAMC, *What’s on the MCAT Exam?*, © 2020, 111 pp.)
**Compared against:** `mcat_topics.json` (749 entries; UTF-8 BOM).
**Phase:** REPORT ONLY — `mcat_topics.json` was **not** modified. Apply step is a separate Phase 2.

---

## Summary

| Metric | Value |
|---|---|
| Science content categories audited | **31** (1A–10A) |
| CARS | excluded — skills section, no factual subtopic outline (see notes) |
| AAMC outline bullet-lines parsed | **1412** across the 31 categories |
| Proposed subtopic additions (Task B) | **8** total — 3 high, 5 medium, 0 low |
| Missing whole topics (Task C) | **2** |

### Headline findings

1. **The JSON is a high-fidelity copy of the AAMC outline.** 13 of 31 categories match the outline *exactly* at the bullet level; the remainder differ only by accidental drops, rewordings, or grouping. The audit surfaced **few** strictly-traceable gaps — the headline below, plus two accidental single-bullet drops.
2. **Biggest gap — 3A “Nervous System: Structure and Function.”** An entire AAMC topic group is absent from the JSON: organization of the vertebrate nervous system, sensor/effector neurons, autonomic (sympathetic/parasympathetic) control, reflexes/reflex arc, and spinal-cord integration. JSON 3A only covers the *cellular* nerve content plus (mislabeled) endocrine content.
3. **⚠ The motivating example (Lineweaver-Burk) is NOT in the AAMC PDF.** The strings `lineweaver`, `double-reciprocal`, and `double reciprocal` appear **nowhere** in the outline. Under the audit’s own ground-truth rule (“every addition must be traceable to specific PDF text”), Lineweaver-Burk is therefore **out of scope** and is not proposed. See **Scope & the ground-truth tension** below.
4. **Data-quality issues exist beyond missing subtopics** (column-bleed text corruption; mislabeled `topic_group` values). These are arguably higher-impact than the missing subtopics but are out of the strict audit scope; they are catalogued in **Appendix B** for awareness.

---

## Scope & the ground-truth tension (read this)

The prompt frames the goal as “the JSON omits canonical high-yield subtopics the AAMC actually tests” and cites **Lineweaver-Burk** as a confirmed example. But the prompt’s **ground-truth rule (critical)** says every proposal must quote specific AAMC outline text and forbids additions “from general MCAT knowledge.”

These two pull in opposite directions, because **the AAMC outline itself is terse.** Verified facts:

- `lineweaver` / `double-reciprocal` → **not found** anywhere in the 111-page PDF.
- The AAMC enzyme-kinetics outline (1A and 5E) literally lists only: *Kinetics — General (catalysis), Michaelis-Menten, Cooperativity, Effects of local conditions*. No Lineweaver-Burk, no Eadie-Hofstee.
- The JSON **already contains** every one of those listed bullets (e.g. `BB_1A_019`).

**Conclusion:** the JSON’s “terseness” largely mirrors the **outline’s** terseness. Honoring the ground-truth rule, this report proposes **only** content quotable from the PDF. Canonical-but-unlisted items (Lineweaver-Burk, Henderson-Hasselbalch, etc.) are deliberately **excluded**. If you want those, that is a different job with **looser criteria** — recommended as a separate pass (see **Recommendations**).

---

## Task A — AAMC outline parse

- **Categories parsed:** 31 science content categories, 1A through 10A, keyed by content-category code. Each maps 1:1 to a `content_category` prefix in the JSON.
- **Outline bullet-lines parsed:** 1412 (lines beginning with ▪ / o / •).
- **Method:** text extracted with PyMuPDF (`fitz`) to UTF-8; split into category blocks on header lines matching `^\d{1,2}[A-E]:`; per-category “dossiers” built pairing the verbatim AAMC block with the JSON entries for that category; gaps detected by a normalized string-diff (every AAMC bullet vs. every JSON topic/subtopic), then **every flagged candidate was adjudicated by hand** against the dossier.

**Parsing caveats / could-not-parse-cleanly:**

- The PDF is a **two-column layout**; `fitz` interleaves the left-column descriptive prose with the right-column topic outline. Category blocks were bounded by headers so prose never crosses a category boundary, but the descriptive prose remains mixed in (it is ignored for gap detection — only ▪/o/• bullets and bold topic headers are compared).
- The **Scientific Inquiry & Reasoning Skills (SIRS)** material (pp. 4–16) and the **CARS** section (pp. 101–111) are *not* factual topic outlines; they were excluded. The four SIRS skill names recur in each section’s overview and bled to the edges of the 3B and 5E blocks; these were trimmed/ignored.
- **Bold topic-header auto-detection was noisy** (wrapped headers, discipline-code suffix lines such as `(BIO, BC)`, stray fragments). This affected only the *machine* first-pass; the final Task C list was confirmed by reading the affected dossiers directly.
- **CARS (the JSON’s 32nd `content_category`)**: 3 JSON entries exist but the AAMC CARS section lists no factual subtopics to compare against, so no gap analysis is possible or meaningful for it.

---

## Task B — Proposed subtopic additions (traceable to the PDF)

All 8 proposals below quote AAMC outline text verbatim. Grouped by section → content category.

### Biological and Biochemical Foundations of Living Systems

#### Content Category 2A

| Proposed subtopic | Attach to | Importance | AAMC source (verbatim) |
|---|---|---|---|
| Lysosomes: membrane-bound vesicles containing hydrolytic enzymes | Characteristics of Eukaryotic Cells *(new entry)* | **medium** | “Lysosomes: membrane-bound vesicles containing hydrolytic enzymes” |

- *Lysosomes: membrane-bound vesicles containing hydrolytic enz* — AAMC lists Lysosomes as a distinct membrane-bound organelle; JSON jumps Mitochondria (BB_2A_016) -> Endoplasmic reticulum (BB_2A_017), skipping it. AAMC group label is 'Membrane-Bound Organelles and Defining Characteristics of Eukaryotic Cells'.

#### Content Category 3A

| Proposed subtopic | Attach to | Importance | AAMC source (verbatim) |
|---|---|---|---|
| Major functions: high-level control and integration of body systems; adaptive capability to external influences | Nervous System: Structure and Function *(new entry)* | **medium** | “Major functions ... High-level control and integration of body” |
| Organization of vertebrate nervous system | Nervous System: Structure and Function *(new entry)* | **high** | “Organization of vertebrate nervous system” |
| Sensor and effector neurons | Nervous System: Structure and Function *(new entry)* | **medium** | “Sensor and effector neurons” |
| Sympathetic and parasympathetic nervous systems: antagonistic control | Nervous System: Structure and Function *(new entry)* | **high** | “Sympathetic and parasympathetic nervous systems: antagonistic control” |
| Reflexes: feedback loop, reflex arc; role of spinal cord and supraspinal circuits | Nervous System: Structure and Function *(new entry)* | **high** | “Reflexes ... Feedback loop, reflex arc” |
| Integration with endocrine system: feedback control | Nervous System: Structure and Function *(new entry)* | **medium** | “Integration with endocrine system: feedback control” |

- *Major functions: high-level control and integration of body * — Part of the entirely-absent 'Nervous System: Structure and Function' topic (see missing_topics).
- *Organization of vertebrate nervous system* — Part of the entirely-absent 'Nervous System: Structure and Function' topic. CNS/PNS organization is high-yield.
- *Sensor and effector neurons* — Afferent/efferent neurons. Part of the absent 'Nervous System: Structure and Function' topic.
- *Sympathetic and parasympathetic nervous systems: antagonisti* — Autonomic nervous system; high-yield. Part of the absent 'Nervous System: Structure and Function' topic.
- *Reflexes: feedback loop, reflex arc; role of spinal cord and* — Reflex arc is classic high-yield content. Part of the absent 'Nervous System: Structure and Function' topic.
- *Integration with endocrine system: feedback control* — POSSIBLY ALREADY PRESENT - VERIFY: BB_3A_025 has the reciprocal 'Integration with nervous system: feedback control' (under the mislabeled 'Lipids'/endocrine group).

### Chemical and Physical Foundations of Biological Systems

#### Content Category 4A

| Proposed subtopic | Attach to | Importance | AAMC source (verbatim) |
|---|---|---|---|
| Speed, velocity (average and instantaneous) | Translational Motion *(new entry)* | **medium** | “Speed, velocity (average and instantaneous)” |

- *Speed, velocity (average and instantaneous)* — JSON 4A has Units (CP_4A_001), Vectors (_002), Vector addition (_003), Acceleration (_004) but drops the 'Speed, velocity' bullet that sits between Vector addition and Acceleration in the outline.

---

## Task C — Missing whole topics

AAMC topic groups with **no corresponding JSON entry at all** (kept separate from Task B). Each would require a **new `topic_id`**.

### 3A — “Nervous System: Structure and Function”

- **AAMC source:** “Nervous System: Structure and Function (BIO)”
- **Section:** Biological and Biochemical Foundations of Living Systems
- **Note:** No JSON topic_group exists for it. JSON 3A covers only 'Nerve Cell' (cellular neuroscience), 'Electrochemistry', 'Biosignaling', and endocrine content (mislabeled under topic_group 'Lipids'). The entire SYSTEM-level nervous-system outline (organization of vertebrate NS; sensor/effector neurons; autonomic sympathetic/parasympathetic control; reflexes; reflex arc; spinal cord) is absent. Requires a NEW topic_id (or several). Its 6 subtopics are itemized in subtopic_additions. Highest-impact finding of this audit.

### 1C — “Evidence That DNA Is Genetic Material”

- **AAMC source:** “Evidence That DNA Is Genetic Material (BIO)”
- **Section:** Biological and Biochemical Foundations of Living Systems
- **Note:** Appears in the AAMC outline as a bare topic header with NO subtopic bullets; no corresponding JSON entry exists (JSON 1C has Mendelian Concepts, Variability, Analytic Methods, Evolution). Would need a new topic_id. Low yield (classic experiments: Griffith; Avery-MacLeod-McCarty; Hershey-Chase).

---

## What was checked and found COMPLETE

To make the negative space explicit: the following 13 categories matched the AAMC outline **exactly at the bullet level** (every ▪/o/• bullet has a JSON counterpart) — no additions proposed:

> **2B, 4B, 4E, 5B, 6A, 6C, 7A, 7B, 8B, 8C, 9A, 9B, 10A**

The remaining categories had only **false-positive** candidates from the string-diff — concepts that ARE present but reworded or merged. Representative examples (all **present**, not proposed):

| Category | String-diff flagged | Actually in JSON as | Verdict |
|---|---|---|---|
| 1A | Motors | BB_1A_010 “Motor proteins in cellular movement” | present (reworded) |
| 1C | Gene / Locus | BB_1C_002 / BB_1C_003 | present (reworded) |
| 2A | Osmosis, Desmosomes, Steroids, organelles | BB_2A_004/007/014/015/016… | present (some garbled) |
| 2C | Ovum and sperm | sub-bullets folded into BB_2C_008 (gametogenesis) | present (folded) |
| 4C | Meters | CP_4C_011 “Ammeters and voltmeters in circuits” | present (reworded) |
| 4D | Pitch | CP_4D_006 “Pitch and frequency of sound waves” | present |
| 5A | Ionization of water | BB? CP_5A_001 (Kw = [H+][OH–]) | present (garbled) |
| 5D | Steroids, Binding, Motor, Reduction | CP_5D_025/015/017/047 | present (reworded) |
| 3B | Specialized Cell — Muscle Cell | BB_3B_061–066 (sarcomeres, troponin…) | present (folded into Muscle System) |
| 6B | Dreaming | PS_6B_012 “Dreaming and states of consciousness” | present |
| 7C | Modeling | PS_7C_005 “Modeling in observational learning” | present |

---

## Recommendations

1. **Apply the 8 Task-B additions + 2 Task-C new topics** in Phase 2 (consume `json_enrichment_proposals.json`). The 3A nervous-system topic is the priority.
2. **Verify** the single flagged item `3A / Integration with endocrine system: feedback control` against `BB_3A_025` before adding (may be a duplicate).
3. **Separate, higher-value follow-up (out of this audit’s scope):** a **data-quality cleanup pass** for the column-bleed corruption and mislabeled `topic_group`s in Appendix B — these likely hurt generation quality more than the few missing subtopics.
4. **Optional canonical-enrichment pass** with *relaxed* criteria (a named, vetted high-yield list rather than strict PDF-tracing) if you want Lineweaver-Burk-class content that the AAMC tests but does not enumerate. This must be a deliberate, separate decision — it contradicts this audit’s ground-truth rule.

---

## Appendix A — Per-category coverage at a glance

| Cat | Section | JSON entries | AAMC bullet-lines | Result |
|---|---|---:|---:|---|
| 1A | Biological | 22 | 50 | complete / present |
| 1B | Biological | 59 | 61 | complete / present |
| 1C | Biological | 29 | 50 | **1 missing topic** |
| 1D | Biological | 44 | 68 | complete / present |
| 2A | Biological | 27 | 50 | **1 subtopic** (Lysosomes) |
| 2B | Biological | 28 | 36 | complete / present |
| 2C | Biological | 24 | 41 | complete / present |
| 3A | Biological | 26 | 42 | **1 missing topic + 6 subtopics** |
| 3B | Biological | 77 | 206 | complete / present |
| 4A | Chemical | 21 | 24 | **1 subtopic** (Speed/velocity) |
| 4B | Chemical | 18 | 29 | complete / present |
| 4C | Chemical | 19 | 43 | complete / present |
| 4D | Chemical | 29 | 45 | complete / present |
| 4E | Chemical | 38 | 52 | complete / present |
| 5A | Chemical | 14 | 26 | complete / present |
| 5B | Chemical | 13 | 27 | complete / present |
| 5C | Chemical | 10 | 16 | complete / present |
| 5D | Chemical | 53 | 101 | complete / present |
| 5E | Chemical | 32 | 73 | complete / present |
| 6A | Psychological | 15 | 28 | complete / present |
| 6B | Psychological | 22 | 54 | complete / present |
| 6C | Psychological | 8 | 24 | complete / present |
| 7A | Psychological | 19 | 72 | complete / present |
| 7B | Psychological | 14 | 18 | complete / present |
| 7C | Psychological | 10 | 23 | complete / present |
| 8A | Psychological | 5 | 7 | complete / present |
| 8B | Psychological | 9 | 13 | complete / present |
| 8C | Psychological | 20 | 39 | complete / present |
| 9A | Psychological | 22 | 40 | complete / present |
| 9B | Psychological | 11 | 35 | complete / present |
| 10A | Psychological | 8 | 19 | complete / present |

---

## Appendix B — Data-quality observations (OUT OF SCOPE; not in proposals.json)

These are **not** missing subtopics (the content is present), so they are excluded from the machine-readable proposals. They are recorded here because they likely degrade downstream generation more than the gaps above.

### B1. Column-bleed text corruption inside JSON strings

The two-column PDF was evidently scraped without column separation when the JSON was first built, so descriptive prose is fused into many `topic`/`subtopics` strings. Examples (verbatim from the JSON):

- `BB_2A_016` topic = “**Mitochondria** for purposes of recognition of ‘self’ versus foreign” (mitochondria text + immune-recognition prose).
- `BB_2A_007` subtopic = “**Osmosis** Cell membranes separate the internal environment of”.
- `BB_3A_016` subtopic = “Terpenes and terpenoids **Endocrine System: Hormones and Their Sources (BIO)**” (a header bled into a subtopic).
- `CP_4A_004` topic = “**Acceleration** acting on them are zero. Many aspects of motion can”.
- `CP_5A_001` subtopics fuse Kw/pH definitions with unrelated “water is unique…” prose.

Affected categories include at least **2A, 2C, 3A, 4A, 5A** (and likely others). A targeted re-scrape or regex cleanup would fix most of these.

### B2. Mislabeled `topic_group` values (content present, grouping wrong)

| Category | JSON entries | Mislabeled `topic_group` | Should be (AAMC) |
|---|---|---|---|
| 3A | BB_3A_017 – BB_3A_026 | “Lipids” | “Endocrine System: Hormones…” / “…Mechanisms of Hormone Action” |
| 4C | CP_4C_018, CP_4C_019 | “Electrochemistry” | “Specialized Cell — Nerve Cell” |
| 5B | all 13 entries | “Covalent Bond” | also “Liquid Phase — Intermolecular Forces” etc. |
| 5D | CP_5D_053 | “Phenols” | “Polycyclic and Heterocyclic Aromatic Compounds” |
| 3B | BB_3B_061 – 066 | “Muscle System” | (folded) “Specialized Cell — Muscle Cell” |

These are grouping/labeling issues, not missing content, so Phase 2 may treat them as an optional cleanup.
