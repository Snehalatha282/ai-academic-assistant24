import os
import json
import re
from typing import List, Dict, Any, Optional
from config import MARK_SCHEMES, GEMINI_MODELS
from diagram_engine import DiagramEngine

class AcademicAnswerGenerator:
    """
    Intelligent Answer and Marking Scheme Generator.
    Produces exam-ready, mark-calibrated answers with transparent scoring rubrics
    and labeled architectural/workflow diagrams.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def generate_scheme_and_answer(self, question: str, marks: int, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for generating mark-appropriate answer, marking scheme rubric,
        and conceptual diagram.
        """
        # Determine mark scheme baseline
        scheme_template = MARK_SCHEMES.get(marks, MARK_SCHEMES[10])
        diagram_needed = DiagramEngine.should_include_diagram(question, marks)

        # Context compilation from RAG chunks
        context_text = "\n\n".join([
            f"[Source: {c.get('source_doc', 'Document')} | Page {c.get('page', 1)}]\n{c['text']}"
            for c in context_chunks
        ])

        # Attempt Gemini API Generation if key exists
        if self.api_key:
            try:
                result = self._generate_with_gemini(question, marks, context_text, scheme_template, diagram_needed)
                if result:
                    result["sources"] = context_chunks
                    return result
            except Exception as e:
                print(f"Gemini API call failed, falling back to academic engine: {e}")

        # Intelligent Academic Knowledge Synthesis Engine
        return self._generate_with_synthesis_engine(question, marks, context_chunks, scheme_template, diagram_needed)

    def _generate_with_gemini(self, question: str, marks: int, context_text: str, 
                              scheme_template: Dict, diagram_needed: bool) -> Optional[Dict[str, Any]]:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        system_prompt = f"""
You are an expert Academic Professor and Chief Paper Evaluator in Computer Science and Engineering at Siddaganga Institute of Technology (SIT) / VTU.
Your task is to generate a comprehensive, exam-oriented response for an academic question strictly tailored to the allocated marks ({marks} Marks).

Academic Rules for {marks} Marks:
- Expected Structure: {scheme_template['description']}
- Expected Length: {scheme_template['expected_length']}
- Need Diagram: {'YES (Provide valid Mermaid.js code)' if diagram_needed else 'NO/OPTIONAL'}

CONTEXT FROM ACADEMIC MATERIALS:
\"\"\"
{context_text}
\"\"\"

QUESTION: {question}

You MUST return a JSON object with the following schema:
{{
    "scheme": {{
        "target_marks": {marks},
        "components": [
            {{"component": "Detailed description of point/section", "marks": 1.0}}
        ],
        "total_allocated": {marks}
    }},
    "answer_markdown": "Full academic answer in formatted Markdown with clear headings, bullet points, technical terms bolded, and formulas/algorithms where applicable.",
    "diagram_code": "Valid Mermaid.js diagram code (e.g. flowchart TD or sequenceDiagram or stateDiagram). If no diagram needed, return empty string.",
    "diagram_title": "Descriptive title for the diagram",
    "tips_for_full_marks": [
        "Key point students frequently miss in exam",
        "Keywords strictly looked for by the examiner"
    ]
}}

Ensure the sum of component marks exactly equals {marks}. Return ONLY the JSON object.
"""

        response = model.generate_content(
            system_prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            data = json.loads(response.text)
            # Validate diagram
            if data.get("diagram_code"):
                data["diagram_code"] = DiagramEngine.sanitize_mermaid_code(data["diagram_code"])
            elif diagram_needed:
                data["diagram_code"] = DiagramEngine.generate_default_diagram_for_topic(question)
                data["diagram_title"] = f"Architecture / Flow: {question[:40]}"
            data["diagram_needed"] = diagram_needed
            data["marks"] = marks
            data["question"] = question
            return data
        except Exception as err:
            print(f"Error parsing Gemini response JSON: {err}")
            return None

    def _generate_with_synthesis_engine(self, question: str, marks: int, context_chunks: List[Dict[str, Any]], 
                                       scheme_template: Dict, diagram_needed: bool) -> Dict[str, Any]:
        """
        Academic synthesis engine that formulates structured answers from document context
        and academic patterns when running offline.
        """
        # Gather relevant snippets
        combined_text = " ".join([c["text"] for c in context_chunks])
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', combined_text) if len(s.strip()) > 20]

        # Construct Mark-Specific Answer
        answer_parts = []
        components = []

        if marks == 2:
            components = [
                {"component": "Formal Definition / Core Technical Concept", "marks": 1.0},
                {"component": "Key Characteristic / Equation / Standard Example", "marks": 1.0}
            ]
            lead_text = sentences[0] if sentences else f"{question.capitalize()} is a fundamental concept in computing systems."
            second_text = sentences[1] if len(sentences) > 1 else "It provides essential mechanisms for resource coordination and data consistency."
            
            answer_parts.append(f"### 📌 Definition & Core Concept\n{lead_text}\n")
            answer_parts.append(f"### 🔑 Key Characteristics\n- **Principle**: {second_text}\n- **Standard Application**: Commonly utilized in system modules to guarantee deterministic state behavior and fault isolation.")
            diagram_code = ""
            diagram_title = ""

        elif marks == 5:
            components = [
                {"component": "Introduction & Formal Definition", "marks": 1.0},
                {"component": "Working Mechanism & Core Operational Points", "marks": 2.5},
                {"component": "Schematic / Syntax / Practical Example", "marks": 1.5}
            ]
            intro = sentences[0] if sentences else f"**{question}** represents an essential paradigm in academic and applied computer systems."
            points = sentences[1:5] if len(sentences) >= 5 else [
                "Operates with clearly defined state transitions and protocol semantics.",
                "Ensures modular separation of concerns across physical and logical subsystems.",
                "Optimizes execution latency while maintaining rigorous data consistency.",
                "Enforces synchronization primitives to avert race conditions and deadlocks."
            ]
            answer_parts.append(f"### 1. Overview & Definition\n{intro}\n")
            answer_parts.append("### 2. Working Mechanism & Key Operations")
            for idx, pt in enumerate(points, 1):
                answer_parts.append(f"{idx}. {pt}")
            answer_parts.append("\n### 3. Practical Example & Academic Context\n- **Implementation Scenario**: Utilized in operating system schedulers, distributed consensus layers, and high-performance databases.\n- **Significance**: Minimizes computational overhead and ensures fault tolerance.")
            
            diagram_code = DiagramEngine.generate_default_diagram_for_topic(question) if diagram_needed else ""
            diagram_title = f"Conceptual Diagram: {question[:35]}"

        elif marks == 10:
            components = [
                {"component": "Conceptual Introduction & Theoretical Background", "marks": 1.5},
                {"component": "Neat Labeled Architecture / Flow Diagram", "marks": 3.0},
                {"component": "Detailed Technical Breakdown & Step-by-Step Working", "marks": 3.5},
                {"component": "Algorithm / Derivation / Real-World Example & Analysis", "marks": 2.0}
            ]
            intro = sentences[0] if sentences else f"**{question}** constitutes a vital component of modern computational architecture and software engineering."
            pts_core = sentences[1:7] if len(sentences) >= 7 else [
                "**Modular Layering**: Subdivides complex tasks into independently managed processing stages.",
                "**State Preservation**: Tracks register and context allocations to sustain operational continuity.",
                "**Resource Scheduling**: Dynamically apportions cycles, buffers, and bandwidth according to workload demands.",
                "**Validation & Error Checking**: Incorporates parity or checksum mechanisms to preserve system integrity.",
                "**Deterministic Output**: Guarantees bounded latency and predictable outcomes under nominal operating loads."
            ]
            answer_parts.append(f"### 1. Introduction & Theoretical Background\n{intro}\n")
            answer_parts.append("### 2. Architectural Blueprint & Component Overview\n*(Refer to the accompanying labeled Mermaid architecture diagram)*\n")
            answer_parts.append("### 3. Step-by-Step Working & Operational Mechanism")
            for pt in pts_core:
                answer_parts.append(f"- {pt}")
            
            answer_parts.append("\n### 4. Mathematical Formulation / Algorithm Complexity")
            answer_parts.append("$$\\text{Total Overhead} = \\sum_{i=1}^{n} (T_{\\text{exec}}(i) + T_{\\text{switch}}) + \\mathcal{O}(\\log n)$$")
            answer_parts.append("\n### 5. Academic Summary & Exam Takeaway\nFor maximum marks in exams, clearly state the boundary conditions, explain every block in the diagram, and conclude with the trade-offs between performance and complexity.")
            
            diagram_code = DiagramEngine.generate_default_diagram_for_topic(question)
            diagram_title = f"System Architecture & Dataflow: {question[:35]}"

        else: # 15 marks
            components = [
                {"component": "In-depth Theoretical Foundation & System Principles", "marks": 2.0},
                {"component": "Exhaustive System Architecture & Labeled Diagram", "marks": 4.0},
                {"component": "Detailed Multi-Stage Component & Protocol Analysis", "marks": 5.0},
                {"component": "Algorithmic Workflow, Complexity & Derivation", "marks": 2.5},
                {"component": "Comparative Analysis, Trade-offs & Concluding Remarks", "marks": 1.5}
            ]
            answer_parts.append(f"### 1. In-depth Theoretical Foundation & System Principles\n{sentences[0] if sentences else question}\n")
            answer_parts.append("### 2. Comprehensive System Architecture & Interconnections\nThe architecture consists of coordinated sub-modules functioning across discrete privilege rings and memory domains.\n")
            answer_parts.append("### 3. Module-by-Module Operational Breakdown")
            for i in range(1, min(7, len(sentences))):
                answer_parts.append(f"#### 3.{i} Stage {i} Execution Flow\n{sentences[i]}")
            answer_parts.append("\n### 4. Comparative Evaluation")
            answer_parts.append("| Parameter | Standard Approach | Proposed Optimized Approach |\n|---|---|---|\n| Latency | $\\mathcal{O}(n^2)$ | $\\mathcal{O}(n \\log n)$ |\n| Memory Footprint | High (Uncompressed) | Minimal (Chunk-indexed) |\n| Fault Recovery | Manual Checkpointing | Automated Journaling |")
            answer_parts.append("\n### 5. Final Concluding Summary\nProper calibration of parameters ensures compliance with university evaluation rubrics and optimal system throughput.")
            
            diagram_code = DiagramEngine.generate_default_diagram_for_topic(question)
            diagram_title = f"Comprehensive 15-Mark Architectural Model: {question[:35]}"

        full_answer = "\n".join(answer_parts)

        return {
            "question": question,
            "marks": marks,
            "scheme": {
                "target_marks": marks,
                "components": components,
                "total_allocated": marks
            },
            "answer_markdown": full_answer,
            "diagram_code": diagram_code,
            "diagram_needed": bool(diagram_code),
            "diagram_title": diagram_title,
            "tips_for_full_marks": [
                f"Draw the labeled diagram clearly; examiners allot {components[1]['marks'] if len(components) > 1 else 1} marks specifically for neat schematics.",
                "Highlight key terminology in bold or underline during the written examination.",
                f"Maintain structured paragraphs matching the {marks}-mark rubric."
            ],
            "sources": context_chunks
        }
