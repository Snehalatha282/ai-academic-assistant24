import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "academic_assistant.db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Project Information
PROJECT_INFO = {
    "institution": "Siddaganga Institute of Technology - Tumakuru",
    "department": "Department of Computer Science and Engineering",
    "title": "AI Academic Assistant: Intelligent PDF Analysis and Scheme-Based Answer Generation",
    "academic_year": "2025-26",
    "team": [
        {"name": "Shamitha H S", "usn": "1SI24CS161", "mobile": "7975348582"},
        {"name": "Skarzang Lamo", "usn": "1SI24CS172", "mobile": "7006820598"},
        {"name": "Snehalatha", "usn": "1SI24CS174", "mobile": "8073815036"},
        {"name": "Sharanya H M", "usn": "1SI24CS213", "mobile": "8310968958"}
    ]
}

# Marking Scheme Specifications according to SIT/VTU guidelines
MARK_SCHEMES = {
    2: {
        "label": "2 Marks (Short Answer)",
        "expected_length": "30 - 60 words (3-5 sentences / bullet points)",
        "breakdown": [
            {"component": "Crisp Definition / Statement / Principle", "marks": 1.0},
            {"component": "Key Feature, Formula, or Standard Example", "marks": 1.0}
        ],
        "components": [
            {"component": "Crisp Definition / Statement / Principle", "marks": 1.0},
            {"component": "Key Feature, Formula, or Standard Example", "marks": 1.0}
        ],
        "diagram_required": False,
        "description": "Precise, conceptual definition or core equation without unnecessary filler."
    },
    5: {
        "label": "5 Marks (Medium Answer)",
        "expected_length": "120 - 200 words (structured paragraphs or 4-6 points)",
        "breakdown": [
            {"component": "Introduction & Definition", "marks": 1.0},
            {"component": "Core Explanation / Working Principle (3-4 points)", "marks": 2.5},
            {"component": "Schematic / Syntax / Formula / Example", "marks": 1.5}
        ],
        "components": [
            {"component": "Introduction & Definition", "marks": 1.0},
            {"component": "Core Explanation / Working Principle (3-4 points)", "marks": 2.5},
            {"component": "Schematic / Syntax / Formula / Example", "marks": 1.5}
        ],
        "diagram_required": "Optional / Recommended if architectural or schematic",
        "description": "Structured explanation covering fundamentals, working, and a relevant example or simple block diagram."
    },
    10: {
        "label": "10 Marks (Long Answer)",
        "expected_length": "350 - 550 words (comprehensive technical essay)",
        "breakdown": [
            {"component": "Introduction & Conceptual Background", "marks": 1.5},
            {"component": "Neat Labeled Diagram / Architecture / Flowchart", "marks": 3.0},
            {"component": "Detailed Working / Algorithm / Derivation / Mechanism", "marks": 3.5},
            {"component": "Real-world Example, Pros/Cons or Complexity Analysis", "marks": 2.0}
        ],
        "components": [
            {"component": "Introduction & Conceptual Background", "marks": 1.5},
            {"component": "Neat Labeled Diagram / Architecture / Flowchart", "marks": 3.0},
            {"component": "Detailed Working / Algorithm / Derivation / Mechanism", "marks": 3.5},
            {"component": "Real-world Example, Pros/Cons or Complexity Analysis", "marks": 2.0}
        ],
        "diagram_required": True,
        "description": "Comprehensive academic answer with neat labeled diagram, in-depth technical explanation, and analytical insights."
    },
    15: {
        "label": "15 Marks (Comprehensive / Essay Answer)",
        "expected_length": "600 - 900 words (thorough multi-section treatise)",
        "breakdown": [
            {"component": "Introduction & Theoretical Foundation", "marks": 2.0},
            {"component": "Comprehensive Architecture & Labeled Diagram", "marks": 4.0},
            {"component": "In-depth Component Analysis & Operation Workflow", "marks": 5.0},
            {"component": "Mathematical Formulation / Algorithm & Code / Case Study", "marks": 2.5},
            {"component": "Comparison Table, Advantages, Limitations & Summary", "marks": 1.5}
        ],
        "components": [
            {"component": "Introduction & Theoretical Foundation", "marks": 2.0},
            {"component": "Comprehensive Architecture & Labeled Diagram", "marks": 4.0},
            {"component": "In-depth Component Analysis & Operation Workflow", "marks": 5.0},
            {"component": "Mathematical Formulation / Algorithm & Code / Case Study", "marks": 2.5},
            {"component": "Comparison Table, Advantages, Limitations & Summary", "marks": 1.5}
        ],
        "diagram_required": True,
        "description": "Exhaustive multi-part answer for full-module coverage, rigorous diagrams, algorithms, and comparative analysis."
    }
}

def calibrate_scheme_components(components_list: list, target_marks: float) -> list:
    """
    Calibrates component allocated marks proportionally so they strictly divide
    and sum up to target_marks exactly.
    """
    if not components_list:
        return []

    target_marks = float(target_marks)
    current_marks = []
    for c in components_list:
        try:
            m = float(c.get("marks", 1.0))
        except (ValueError, TypeError):
            m = 1.0
        current_marks.append(max(0.1, m))

    total_current = sum(current_marks)
    if total_current <= 0:
        total_current = float(len(components_list))
        current_marks = [1.0] * len(components_list)

    # Scale proportionally
    scaled_marks = [round((m / total_current) * target_marks, 1) for m in current_marks]

    # Adjust rounding remainder on the largest element
    diff = round(target_marks - sum(scaled_marks), 1)
    if diff != 0:
        max_idx = scaled_marks.index(max(scaled_marks))
        scaled_marks[max_idx] = round(scaled_marks[max_idx] + diff, 1)

    calibrated = []
    for c, sm in zip(components_list, scaled_marks):
        comp_copy = dict(c)
        comp_copy["marks"] = sm
        calibrated.append(comp_copy)

    return calibrated

# Supported LLM Models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash"
]
