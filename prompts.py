"""
All prompts and structured knowledge used by the pipeline.

Keeping prompts in one file makes them easy to review, version-control, and
tweak without touching any pipeline logic.
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Semantic label scheme (A–E)
# ---------------------------------------------------------------------------

SEMANTIC_LABELS = ["A", "B", "C", "D", "E"]

# ---------------------------------------------------------------------------
# CHECK_LIST — questions for the Extraction stage (fact "dehydration")
#
# Each label maps to a dict of {display_title: question_for_LLM}.
# The QA agent answers every question against the relevant text slices.
# ---------------------------------------------------------------------------

CHECK_LIST: Dict[str, Dict[str, str]] = {
    "A": {
        "Title of the Medical Guideline":
            "What is the title of the medical guideline?",
        "Organization/Study group":
            "Could you please provide the names of all the organizations or expert groups that launched the guideline (for example: ESC [European Society of Cardiology])?",
        "DOI":
            "What is the DOI of the Medical Guideline, starting with `https://doi.org/` (for example: https://doi.org/10.1093/eurheartj/ehae178)?",
        "Publication year and version":
            "What is the publication year and version?",
    },
    "B": {
        "Target condition definition":
            "Could you please extract the definition of the target condition by identifying phrases such as `definition of [target condition]`?",
        "Health problems related to the condition":
            "What are the health problems related to the target condition, including terms such as health problems, related conditions, psychological impact?",
        "Risk factors":
            "What are the risk factors associated with the target condition, including but not limited to age, lifestyle, medical diseases, immunity status, environmental exposures, certain medications, and psychological stress?",
        "Symptoms":
            "What are the symptoms of the target condition, defined as subjective evidence of disease or physical disturbance observed by the patient?",
        "Subtypes":
            "What are the subtypes of the target condition?",
        "Complications":
            "What are the complications of the target condition that can be identified by terms like `complications`?",
        "Staging and progression":
            "What is the staging (progress) of the condition, and how will it progress?",
        "Epidemiology":
            "What are the key epidemiological characteristics of the target condition, including its prevalence, incidence, morbidity, and mortality rates, along with associated problems, susceptible regions, and age group distribution patterns?",
        "Duration":
            "What is the expected duration of the condition?",
    },
    "C": {
        "Treatment and intervention recommendations":
            "Extract ALL recommendations related to treatments and interventions, including options, benefits/harms, evidence status, target population, and when to use or not use. Format as a structured list. If none found, return NOTFOUND.",
        "Patient support and self-management":
            "Extract ALL recommendations related to patient support and self-management, including lifestyle interventions, patient education, and treatment adherence guidance. Format as a structured list. If none found, return NOTFOUND.",
        "Patient-provider communication and care coordination":
            "Extract ALL recommendations related to patient-provider communication and care coordination, including guidance for conversations, care overuse/underuse, and implementation barriers. Format as a structured list. If none found, return NOTFOUND.",
    },
    "D": {
        "Purpose of the PVG":
            "What is the purpose of the Clinical guideline?",
        "Intended use of the PVG":
            "What is the intended use of the Clinical guideline?",
        "Target users":
            "Who are the target users of the Clinical guideline?",
    },
    "E": {
        "Terms and abbreviations":
            "Could you please extract the list of terms and abbreviations included in the CPG?",
        "Funding sources":
            "What are the funding source(s) of the CPG, and did they have any role or influence in the guideline development process?",
        "Conflicts of interests":
            "Could you elaborate on the conflicts of interest among contributors to the CPG and how these conflicts were managed?",
    },
}

# ---------------------------------------------------------------------------
# QA system prompt (used by the extraction agent)
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """\
You are a medical question-answering assistant. You will receive **a question** \
and a **text passage** (which may contain structural tags like [TITLE]).
Your task is to extract medical facts from the passage to answer the question.

Follow these rules:
1. **Focus on content:** Ignore any structural markers like [TITLE] or ### CONTEXT. \
Only extract the underlying medical information.
2. **Strictness:** If the passage definitely does not contain relevant information, \
respond with `NOTFOUND`.
3. **Completeness:** If information is found, provide a comprehensive answer using \
only details from the passage.
4. **No commentary:** Output ONLY the answer or `NOTFOUND`. No explanations.
"""

# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------

FACTS_SIDEBAR_INSTRUCTION = """
### GLOBAL KNOWLEDGE BASE (Facts Sidebar)
You are provided with a 'Facts Sidebar' which contains structured medical facts \
extracted from the entire guideline (Labels A-E).
- USE THIS for cross-label context (e.g., use disease definitions from B when \
writing the Treatment section C).
- DO NOT repeat the background information verbatim; use it to ensure logical \
consistency.
"""

STREAM_BIN_PROMPT = """\
You are a Senior Medical Editor rewriting a specific chapter of a clinical \
guideline into a "Patient Version Guide" (PVG).

### YOUR ROLE
You are a warm, supportive, and professional health educator. Your goal is to \
empower patients by making complex clinical data understandable without losing \
accuracy.

### INPUT CONTEXT
- **Current Chapter/Section**: {context_hint}
- **Global Facts Sidebar**: {facts}
- **Original Source Content**: (See below)

### REWRITING RULES
1. **Q&A Narrative Style**: Frame headings as questions patients actually ask \
(e.g., "What is Stage 2?", "What are my surgical options?").
2. **Sequential Flow**: Strictly follow the order of the source content.
3. **Tone**: Use "your doctor", "your care team", and "you". Avoid "patients" \
or "clinicians".
4. **Information Density**:
   - Explain technical terms the first time they appear.
   - Omit technical research methodology, p-values, or sample sizes.

### VISUAL & NAVIGATION MANDATES
1. At the end of each logical sub-chapter, include a brief "Questions to Ask \
Your Doctor" list.
2. If a decision process or classification is described, draw a **Mermaid flowchart**.
3. If comparing drugs or side effects, use a **Markdown Table**.
4. **CRITICAL**: Enclose ALL Mermaid node labels in double quotes to prevent \
syntax errors.

### OUTPUT FORMAT
Return the rewritten Markdown directly. Ensure clear spacing and professional \
formatting.
"""

# ---------------------------------------------------------------------------
# Title-hierarchy agent prompt
# ---------------------------------------------------------------------------

TITLE_HIERARCHY_SYSTEM = """\
You are a Senior Medical Document Architect. Reconstruct the logical tree of \
a clinical guideline.

### GLOBAL NAVIGATION MAP (Known Root Chapters):
{history_map}

### CRITICAL LOGIC RULES:
1. **Logical Nesting**: For each item in the CURRENT BATCH, identify its \
logical parent.
2. **Reference History**: You MUST reference IDs from the GLOBAL NAVIGATION \
MAP above if a heading belongs to a previously started chapter.
3. **HARD OVERRIDE RULES**:
   - If title contains 'Words to know', 'Glossary', 'Contributors', or \
'Index' -> Label MUST be **E**, parent_id MUST be null.
   - If title contains 'Tests', 'Diagnosis', 'Workup' -> Label MUST be **D**.
   - If title contains 'Treatment', 'Therapy', 'Management' -> Label MUST \
be **C**.
4. **keep field** — set `keep: false` for sections that are NOT useful for \
patients and should be excluded from the Patient Version Guide:
   - References / Bibliography / Works Cited
   - Author lists / Contributors / Panel Members / Disclosures
   - Table of Contents / Index
   - Version update logs / Change history / "Updates in Version …"
   - Methodology / Conflict of Interest / Acknowledgements
   - Supplementary materials / Appendices
   **ALWAYS `keep: true`** for clinical summary sections such as \
"Key messages", "What to do", "Summary of recommendations", \
"Key points", "Practice points", or any section containing actionable \
clinical recommendations — these are high-value patient content.
   All other clinical content sections MUST have `keep: true`.

### OUTPUT FORMAT:
Respond ONLY with a JSON object: \
{{"items": [{{"id": int, "parent_id": int|null, "label": str, "keep": bool}}]}}
"""

# ---------------------------------------------------------------------------
# Label keyword heuristics (fallback when LLM is unavailable)
# ---------------------------------------------------------------------------

LABEL_KEYWORDS: Dict[str, list] = {
    "A": ["version", "published", "panel", "NCCN",
          "Clinical Practice Guidelines"],
    "B": ["definition", "symptoms", "risk factors", "epidemiology",
          "incidence", "staging", "pathology", "tnm", "grade"],
    "C": ["treatment", "therapy", "management", "recommendation", "surgery",
          "chemotherapy", "radiation", "bcg", "intravesical"],
    "D": ["diagnosis", "test", "testing", "workup", "imaging", "mri",
          "ct scan", "biopsy", "cytology", "metabolic"],
    "E": ["terms", "funding", "conflicts", "interest", "contributors",
          "supplementary", "follow-up", "words to know", "glossary"],
}
