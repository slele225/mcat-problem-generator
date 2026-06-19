# mcat_topics.json Data-Quality Audit — Phase 2b (Report Only)

**Targets:** (A) column-bleed / fused-field corruption, (B) mislabeled `topic_group`.
**Source of truth for reconstruction:** the AAMC outline PDF (`what's on the mcat pdf.pdf`), same as Phase 1.
**Phase:** REPORT ONLY — `mcat_topics.json` was **not** modified (`git diff` confirmed empty).

---
## Summary

| Metric | Value |
|---|---|
| Entries affected by **column-bleed** (Task A) | **35** topic_ids (53 fields) |
| Entries affected by **mislabeled topic_group** (Task B) | **22** topic_ids |
| **Total unique** entries affected | **56** of 749 (7.5%) |
| Existing questions impacted — **production banks** | **75** (bleed) + **68** (mislabel) of 1551 tagged Q |
| Existing questions impacted — **all runs** | 216 (bleed) + 83 (mislabel) of 2851 |
| Proposals requiring a **topic_id change** (HIGH-RISK) | **0** — every fix preserves topic_id |

**Bottom line:** every proposed fix is a **text cleanup** or a **topic_group rename** — *no topic_id changes*, so nothing orphans existing question tags or breaks the weak-topic / guide joins (which key on topic_id). The column-bleed is heavily concentrated in **2A** (24 of 53 fields) — the cell-membrane/organelle block.

---
## How questions are stored (blast-radius reachability)

The local question bank is reachable: generated items live in `runs/<run-name>/*.jsonl`.
- `discrete_questions.jsonl` — each question carries `topic_id`, `content_category`, `topic_group`, `topic`, `subtopics_tested` (denormalized copies of the JSON entry at generation time).
- `science_passages.jsonl` — each passage carries `topic_ids` (list), `content_category`, `topic_group`; questions are nested under `questions`.
- `cars_passages.jsonl` — **no `topic_id`** (CARS is subject-based), so CARS questions are unaffected by this audit.

Because tags are **denormalized into each question**, blast radius is computed locally below. Note two consequences for the apply phase: (1) fixing `mcat_topics.json` does **not** retroactively fix the copies already baked into existing questions — those would need a separate backfill if you care about the stored `topic`/`subtopics_tested`/`topic_group` strings; (2) the join key `topic_id` is preserved by every proposal here, so existing questions stay correctly linked regardless.

If the authoritative bank is the Supabase copy rather than these local runs, run this to get the true counts:

```sql
-- questions tagged to an affected topic_id (routing/steering impact)
SELECT topic_id, count(*) AS n_questions
FROM questions
WHERE topic_id IN (
  'BB_1D_013',
  'BB_1D_040',
  'BB_2A_007',
  'BB_2A_008',
  'BB_2A_009',
  'BB_2A_010',
  'BB_2A_011',
  'BB_2A_012',
  'BB_2A_013',
  'BB_2A_014',
  'BB_2A_015',
  'BB_2A_016',
  'BB_2A_017',
  'BB_2A_018',
  'BB_2B_002',
  'BB_2C_007',
  'BB_2C_008',
  'BB_3A_016',
  'BB_3A_017',
  'BB_3A_018',
  'BB_3A_019',
  'BB_3A_020',
  'BB_3A_021',
  'BB_3A_022',
  'BB_3A_023',
  'BB_3A_024',
  'BB_3A_025',
  'BB_3A_026',
  'BB_3B_061',
  'BB_3B_062',
  'BB_3B_063',
  'BB_3B_064',
  'BB_3B_065',
  'BB_3B_066',
  'CP_4A_001',
  'CP_4A_002',
  'CP_4A_004',
  'CP_4A_005',
  'CP_4C_018',
  'CP_4C_019',
  'CP_4D_011',
  'CP_4D_012',
  'CP_4D_013',
  'CP_4D_014',
  'CP_4D_016',
  'CP_4E_016',
  'CP_5A_001',
  'CP_5A_002',
  'CP_5A_003',
  'CP_5B_011',
  'CP_5B_012',
  'CP_5B_013',
  'CP_5D_052',
  'CP_5D_053',
  'CP_5E_008',
  'PS_8C_007'
)
GROUP BY topic_id
ORDER BY n_questions DESC;
```

---
## Task A — Column-bleed / fused-field corruption

53 fields across 35 topic_ids. Two failure modes from the original 2-column-PDF scrape: **43** fields fuse left-column *description prose* into a topic/subtopic; **10** fuse a *foreign section header*. Confidence: {'high': 49, 'med': 4}. Proposed values are the clean reconstruction (verify the few marked 'med').

### 1D  *(Bio/Biochem)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `BB_1D_013`.topic | high | Polysaccharides Glycolysis, Gluconeogenesis, and the Pentose | Polysaccharides |
| `BB_1D_040`.topic | high | Mitochondria, apoptosis, oxidative stress (BC) Hormonal Regulation and Integration of | Mitochondria, apoptosis, oxidative stress (BC) |

### 2A  *(Bio/Biochem)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `BB_2A_007`.subtopics[1] | high | Osmosis Cell membranes separate the internal environment of | Osmosis |
| `BB_2A_008`.topic | high | Colligative properties; osmotic pressure (GC) the cell from the external environment. The | Colligative properties; osmotic pressure (GC) |
| `BB_2A_008`.subtopics[0] | high | Passive transport specialized structure of the membrane, as described in | Passive transport |
| `BB_2A_008`.subtopics[1] | high | Active transport the fluid mosaic model, allows the cell to be | Active transport |
| `BB_2A_009`.topic | high | Sodium/potassium pump selectively permeable and dynamic, with homeostasis | Sodium/potassium pump |
| `BB_2A_010`.topic | high | Membrane channels maintained by the constant movement of molecules | Membrane channels |
| `BB_2A_011`.topic | high | Membrane potential across the membranes through a combination of | Membrane potential |
| `BB_2A_012`.topic | high | Membrane receptors active and passive processes driven by several forces, | Membrane receptors |
| `BB_2A_013`.topic | high | Exocytosis and endocytosis including electrochemical gradients. | Exocytosis and endocytosis |
| `BB_2A_014`.subtopics[0] | high | Gap junctions Eukaryotic cells also maintain internal membranes | Gap junctions |
| `BB_2A_014`.subtopics[1] | high | Tight junctions that partition the cell into specialized regions. These | Tight junctions |
| `BB_2A_014`.subtopics[2] | high | Desmosomes internal membranes facilitate cellular processes by minimizing conflicting interactions and increasing Membrane-Boun... | Desmosomes |
| `BB_2A_015`.subtopics[0] | high | Compartmentalization, storage of genetic matrix, cells of multicellular organisms organize into information tissues, organs, an... | Compartmentalization, storage of genetic information |
| `BB_2A_015`.subtopics[1] | high | Nucleolus: location and function membrane-associated proteins also play key roles in | Nucleolus: location and function |
| `BB_2A_015`.subtopics[2] | high | Nuclear envelope, nuclear pores identifying tissues or recent events in the cell’s history | Nuclear envelope, nuclear pores |
| `BB_2A_016`.topic | high | Mitochondria for purposes of recognition of “self” versus foreign | Mitochondria |
| `BB_2A_016`.subtopics[2] | high | Self-replication The content in this category covers the composition, | Self-replication |
| `BB_2A_017`.topic | high | Endoplasmic reticulum organelles of eukaryotic cells; and the structure and | Endoplasmic reticulum |
| `BB_2A_017`.subtopics[0] | high | Rough and smooth components function of the major cytoskeletal elements. It covers | Rough and smooth components |
| `BB_2A_017`.subtopics[1] | high | Rough endoplasmic reticulum site of ribosomes the energetics of and mechanisms by which | Rough endoplasmic reticulum site of ribosomes |
| `BB_2A_017`.subtopics[2] | high | Double-membrane structure molecules, or groups of molecules, move across cell | Double-membrane structure |
| `BB_2A_017`.subtopics[3] | high | Role in membrane biosynthesis membranes. It also covers how cell-cell junctions and | Role in membrane biosynthesis |
| `BB_2A_017`.subtopics[4] | high | Role in biosynthesis of secreted proteins the extracellular matrix interact to form tissues with | Role in biosynthesis of secreted proteins |
| `BB_2A_018`.topic | high | Golgi apparatus: general structure and role in packaging and secretion within certain organelles that are different from their ... | Golgi apparatus: general structure and role in packaging and secretion |

### 2C  *(Bio/Biochem)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `BB_2C_007`.topic | high | Oncogenes, apoptosis differentiate into many different types of cells, each | Oncogenes, apoptosis |
| `BB_2C_008`.topic | high | Gametogenesis by meiosis spatial-temporal gradients in the interactions between | Gametogenesis by meiosis |
| `BB_2C_008`.subtopics[0] | high | Differences in formation structural and functional divergence of cells into | Differences in formation |
| `BB_2C_008`.subtopics[1] | high | Differences in morphology specialized structures, organs, and tissues. The | Differences in morphology |
| `BB_2C_008`.subtopics[2] | high | Relative contribution to next generation interaction of stimuli and genes is also explained by | Relative contribution to next generation |

### 2B  *(Bio/Biochem)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `BB_2B_002`.topic | high | Impact on biology Classification and Structure of Prokaryotic Cells (BIO) | Impact on biology |

### 3A  *(Bio/Biochem)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `BB_3A_016`.subtopics[1] | high | Terpenes and terpenoids Endocrine System: Hormones and Their Sources (BIO) | Terpenes and terpenoids |
| `BB_3A_021`.topic | high | Neuroendocrinology ― relation between neurons and hormonal systems Endocrine System: Mechanisms of Hormone Action (BIO) | Neuroendocrinology - relation between neurons and hormonal systems |

### 4A  *(Chem/Phys)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `CP_4A_001`.topic | high | Units and dimensions The motion of any object can be described in terms of | Units and dimensions |
| `CP_4A_002`.topic | high | Vectors, components displacement, velocity, and acceleration. Objects | Vectors, components |
| `CP_4A_004`.topic | high | Acceleration acting on them are zero. Many aspects of motion can | Acceleration |
| `CP_4A_005`.topic | high | Newton’s First Law, inertia different forms. In a living system, the energy for | Newton's First Law, inertia |

### 4D  *(Chem/Phys)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `CP_4D_011`.topic | high | Thin films, diffraction grating, single-slit diffraction When mechanical energy is transmitted through solids, | Thin films, diffraction grating, single-slit diffraction |
| `CP_4D_012`.topic | high | Other diffraction phenomena, X-ray diffraction liquids, and gases, oscillating pressure waves known as | Other diffraction phenomena, X-ray diffraction |
| `CP_4D_013`.topic | high | Polarization of light: linear and circular “sound” are generated. Sound waves are audible if the | Polarization of light: linear and circular |
| `CP_4D_014`.topic | high | Properties of electromagnetic radiation sensory elements of the ear vibrate in response to | Properties of electromagnetic radiation |
| `CP_4D_014`.subtopics[0] | high | Velocity equals constant c, in vacuo exposure to these vibrations. The detection of reflected | Velocity equals constant c, in vacuo |
| `CP_4D_014`.subtopics[1] | med | Electromagnetic radiation consists of sound waves is used in ultrasound imaging. This perpendicularly oscillating electric and ... | Electromagnetic radiation consists of perpendicularly oscillating electric and magnetic fields; direction of propagation is perpendicular to both |
| `CP_4D_016`.topic | high | Visual spectrum, color interact with matter. | Visual spectrum, color |

### 4E  *(Chem/Phys)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `CP_4E_016`.topic | high | Photoelectric effect The Periodic Table ― Classification of Elements | Photoelectric effect |

### 5A  *(Chem/Phys)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `CP_5A_001`.subtopics[0] | med | Kw, its approximate value (Kw = [H+][OH–] = 10–14 properties of water allow it to strongly interact with at 25°C, 1 atm) and mo... | Kw, its approximate value (Kw = [H+][OH-] = 10-14 at 25C, 1 atm) |
| `CP_5A_001`.subtopics[1] | med | Definition of pH: pH of pure water Water is also unique in its ability to absorb energy and ▪ Conjugate acids and bases (e.g., ... | Definition of pH: pH of pure water |
| `CP_5A_002`.topic | high | Strong acids and bases (e.g., nitric, sulfuric) necessary to sustain life. | Strong acids and bases (e.g., nitric, sulfuric) |
| `CP_5A_003`.subtopics[0] | med | Dissociation of weak acids and bases with or The content in this category covers the nature of without added salt solutions, so... | Dissociation of weak acids and bases with or without added salt |

### 5D  *(Chem/Phys)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `CP_5D_052`.topic | high | Oxidation and reduction (e.g., hydroquinones, ubiquinones): biological 2e– redox centers Polycyclic and Heterocyclic Aromatic C... | Oxidation and reduction (e.g., hydroquinones, ubiquinones): biological 2e- redox centers |

### 5E  *(Chem/Phys)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `CP_5E_008`.subtopics[2] | high | Flavoproteins Energy Changes in Chemical Reactions ― | Flavoproteins |

### 8C  *(Psych/Soc)*

| topic_id.field | conf | current (corrupted) | proposed (clean) |
|---|---|---|---|
| `PS_8C_007`.topic | high | Perspectives on bureaucracy (e.g., iron law of oligarchy, McDonaldization) Self-Presentation and Interacting With Others (PSY, ... | Perspectives on bureaucracy (e.g., iron law of oligarchy, McDonaldization) |

---
## Task B — Mislabeled `topic_group`

22 topic_ids. Confidence: {'high': 15, 'medium': 1, 'low': 6}. **Guides-pipeline note:** the guide unit key is `(content_category, topic_group)`, so every rename below re-splits/re-merges a guide unit — these are the changes that most affect guide generation. All are topic_group-only (topic_id preserved).

### 3A  *(Bio/Biochem)*

| topic_id | current_group | -> proposed_group | conf | why |
|---|---|---|---|---|
| `BB_3A_017` | Lipids | Endocrine System: Hormones and Their Sources | high | endocrine content tagged "Lipids" |
| `BB_3A_018` | Lipids | Endocrine System: Hormones and Their Sources | high | endocrine content tagged "Lipids" |
| `BB_3A_019` | Lipids | Endocrine System: Hormones and Their Sources | high | endocrine content tagged "Lipids" |
| `BB_3A_020` | Lipids | Endocrine System: Hormones and Their Sources | high | endocrine content tagged "Lipids" |
| `BB_3A_021` | Lipids | Endocrine System: Hormones and Their Sources | high | endocrine content tagged "Lipids" |
| `BB_3A_022` | Lipids | Endocrine System: Mechanisms of Hormone Action | high | endocrine content tagged "Lipids" |
| `BB_3A_023` | Lipids | Endocrine System: Mechanisms of Hormone Action | high | endocrine content tagged "Lipids" |
| `BB_3A_024` | Lipids | Endocrine System: Mechanisms of Hormone Action | high | endocrine content tagged "Lipids" |
| `BB_3A_025` | Lipids | Endocrine System: Mechanisms of Hormone Action | high | endocrine content tagged "Lipids" |
| `BB_3A_026` | Lipids | Endocrine System: Mechanisms of Hormone Action | high | endocrine content tagged "Lipids" |

### 3B  *(Bio/Biochem)*

| topic_id | current_group | -> proposed_group | conf | why |
|---|---|---|---|---|
| `BB_3B_061` | Muscle System | Specialized Cell - Muscle Cell | low | AAMC separates this from "Muscle System"; folding is arguably acceptable - verify |
| `BB_3B_062` | Muscle System | Specialized Cell - Muscle Cell | low | AAMC separates this from "Muscle System"; folding is arguably acceptable - verify |
| `BB_3B_063` | Muscle System | Specialized Cell - Muscle Cell | low | AAMC separates this from "Muscle System"; folding is arguably acceptable - verify |
| `BB_3B_064` | Muscle System | Specialized Cell - Muscle Cell | low | AAMC separates this from "Muscle System"; folding is arguably acceptable - verify |
| `BB_3B_065` | Muscle System | Specialized Cell - Muscle Cell | low | AAMC separates this from "Muscle System"; folding is arguably acceptable - verify |
| `BB_3B_066` | Muscle System | Specialized Cell - Muscle Cell | low | AAMC separates this from "Muscle System"; folding is arguably acceptable - verify |

### 4C  *(Chem/Phys)*

| topic_id | current_group | -> proposed_group | conf | why |
|---|---|---|---|---|
| `CP_4C_018` | Electrochemistry | Specialized Cell - Nerve Cell | high | nerve-cell content tagged "Electrochemistry" |
| `CP_4C_019` | Electrochemistry | Specialized Cell - Nerve Cell | high | nerve-cell content tagged "Electrochemistry" |

### 5B  *(Chem/Phys)*

| topic_id | current_group | -> proposed_group | conf | why |
|---|---|---|---|---|
| `CP_5B_011` | Covalent Bond | Liquid Phase - Intermolecular Forces | high | IMF content lumped under "Covalent Bond" |
| `CP_5B_012` | Covalent Bond | Liquid Phase - Intermolecular Forces | high | IMF content lumped under "Covalent Bond" |
| `CP_5B_013` | Covalent Bond | Liquid Phase - Intermolecular Forces | high | IMF content lumped under "Covalent Bond" |

### 5D  *(Chem/Phys)*

| topic_id | current_group | -> proposed_group | conf | why |
|---|---|---|---|---|
| `CP_5D_053` | Phenols | Polycyclic and Heterocyclic Aromatic Compounds | medium | heterocyclic content tagged "Phenols" |

> **Note on 3A:** `BB_3A_016` (topic_group 'Lipids', subtopics Steroids/Terpenes) is **correctly** 'Lipids' — AAMC 3A genuinely has a 'Lipids' group. Only `BB_3A_017`–`BB_3A_026` (endocrine content) are mislabeled. Splitting them restores the two real AAMC groups: *Endocrine System: Hormones and Their Sources* and *…Mechanisms of Hormone Action*.

> **Note on 5B:** only the 3 intermolecular-forces entries are misplaced; the stereochemistry entries (`CP_5B_005`–`010`) legitimately sit under 'Covalent Bond' (AAMC lists stereochemistry as a sub-bullet there).

> **Note on 3B (low-confidence):** `BB_3B_061`–`066` hold the AAMC 'Specialized Cell — Muscle Cell' bullets but are folded into 'Muscle System'. AAMC separates them; folding is arguably acceptable. Listed for completeness — verify before acting (this is the only group whose correctness is genuinely debatable).

---
## Task C — Blast radius & severity

### Severity classification

| Risk | Count | Why |
|---|---|---|
| Text cleanup (column-bleed), topic_id preserved | 53 fields | Edits `topic`/`subtopics` strings only. Additive-safe; no orphaning. |
| topic_group rename, topic_id preserved | 22 topic_ids | Re-buckets guide units; existing question links (by topic_id) intact. |
| **topic_id change (HIGH-RISK)** | **0** | None proposed. Any such change would orphan existing question tags. |

### Existing-question blast radius (local `runs/` banks)

Counts = questions tagged to an affected `topic_id`. Test runs are duplicative; the **production banks** are the meaningful figures.

| Bank | type | total Q | on bleed-affected topics | on mislabel-affected topics |
|---|---|--:|--:|--:|
| beta_bank_v1 | discrete | 397 | 0 | 16 |
| beta_bank_v1 | science | 277 | 20 | 12 |
| prod_discrete | discrete | 786 | 32 | 40 |
| prod_science | science | 91 | 23 | 0 |
| **Production subtotal** | | **1551** | **75** | **68** |
| *All runs (incl. test)* | | *2851* | *216* | *83* |

**Interpretation:** in the production banks, ~**75** generated questions were steered by a column-bled (corrupted) topic/subtopic string and ~**68** carry a mislabeled `topic_group`. The bled-steering ones are the quality concern — e.g. any question generated for `BB_2A_016` saw the steering string *"Mitochondria for purposes of recognition of 'self' versus foreign"*, which could push the item toward immune-recognition content rather than mitochondria.

---
## Method & confidence

- PDF text extracted with PyMuPDF (`fitz`) via the **system** Python (`...Python312\python.exe`); the repo `.venv` lacks pip and any PDF lib. `mcat_topics.json` read with `utf-8-sig`.
- Column-bleed detected by matching each `topic`/`subtopics` string against (a) the category's **description-paragraph prose** (runs of >=3 consecutive non-bullet lines) and (b) the set of **AAMC section headers**, then reconstructing the clean head.
- **False positives were aggressively pruned by hand.** An automated first pass flagged 176 then 91 candidates; most were artifacts of AAMC bullets that *wrap across PDF lines* (a bullet's own continuation looked like 'extra' text) or of short header words (e.g. 'transcription', 'consciousness') legitimately recurring inside clean phrases. The 53 reported here are the manually-confirmed definite cases; entries marked **med** need a human to verify the exact reconstruction.
- Per the guardrail, anything that looked merely *odd* (e.g. deliberate rewordings like `BB_1C_002` "Gene as unit of heredity", or clean wrapped bullets) was **excluded**, not proposed.
