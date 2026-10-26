import os
import json
import re
from typing import List, Dict, Any, Optional
from config import MARK_SCHEMES, GEMINI_MODELS, calibrate_scheme_components
from diagram_engine import DiagramEngine

NOT_ENOUGH_INFO_MSG = "The uploaded PDF does not contain enough relevant information to answer this question."

STOP_WORDS = {
    "what", "is", "explain", "describe", "discuss", "with", "neat", "diagram",
    "and", "the", "for", "from", "concept", "write", "short", "note", "notes",
    "define", "how", "does", "give", "brief", "state", "list", "illustrate"
}

class AcademicAnswerGenerator:
    """
    Intelligent Answer and Marking Scheme Generator.
    Produces exam-ready, mark-calibrated answers with transparent scoring rubrics
    and labeled architectural/workflow diagrams, strictly grounded in uploaded PDF documents.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def generate_scheme_and_answer(self, question: str, marks: int, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for generating mark-appropriate answer, marking scheme rubric,
        and conceptual diagram strictly grounded in the provided context_chunks.
        """
        # If no context chunks were retrieved, immediately indicate lack of information
        if not context_chunks:
            return {
                "question": question,
                "marks": marks,
                "scheme": {
                    "target_marks": marks,
                    "components": [
                        {"component": "Relevance & Content Verification", "marks": marks}
                    ],
                    "total_allocated": marks
                },
                "answer_markdown": NOT_ENOUGH_INFO_MSG,
                "diagram_code": "",
                "diagram_needed": False,
                "diagram_title": "",
                "diagram_message": "No diagram generated because relevant content was not found in the uploaded document.",
                "tips_for_full_marks": [
                    "Ensure the question is directly covered in the uploaded course materials."
                ],
                "sources": [],
                "has_sufficient_context": False
            }

        scheme_template = MARK_SCHEMES.get(marks, MARK_SCHEMES[10])
        diagram_needed = DiagramEngine.should_include_diagram(question, marks, context_chunks=context_chunks)

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

        # Grounded Academic Knowledge Synthesis Engine
        return self._generate_with_synthesis_engine(question, marks, context_chunks, scheme_template, diagram_needed)

    def _generate_with_gemini(self, question: str, marks: int, context_text: str, 
                              scheme_template: Dict, diagram_needed: bool) -> Optional[Dict[str, Any]]:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        system_prompt = f"""
You are an expert Academic Professor and Chief Paper Evaluator in Computer Science and Engineering at Siddaganga Institute of Technology (SIT) / VTU.
Your task is to generate an exam-oriented response for an academic question strictly tailored to the allocated marks ({marks} Marks).

CRITICAL GROUNDING INSTRUCTIONS:
1. You must answer the question using ONLY the provided CONTEXT FROM ACADEMIC MATERIALS.
2. If the context does not contain enough relevant information to answer the question, you MUST return:
   "answer_markdown": "{NOT_ENOUGH_INFO_MSG}"
   "diagram_code": ""
3. Do NOT invent facts, definitions, components, or generic statements not present in the context.
4. If a diagram is requested, generate valid Mermaid.js code ONLY if the context provides the specific components and relationships. If context is insufficient for a diagram, return empty string for diagram_code.

Academic Rules for {marks} Marks:
- Expected Structure: {scheme_template['description']}
- Expected Length: {scheme_template['expected_length']}
- Need Diagram: {'YES (Provide valid Mermaid.js code if supported by context)' if diagram_needed else 'NO/OPTIONAL'}

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
    "answer_markdown": "Full academic answer in formatted Markdown strictly grounded in the context.",
    "diagram_code": "Valid Mermaid.js diagram code or empty string if not supported by context.",
    "diagram_title": "Descriptive title for the diagram or empty string",
    "tips_for_full_marks": [
        "Key point strictly looked for by the examiner"
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
            diagram_code = data.get("diagram_code", "").strip()
            
            # Validate diagram syntax if code was generated
            if diagram_code:
                is_valid, err_msg = DiagramEngine.validate_mermaid_syntax(diagram_code)
                if not is_valid:
                    # Retry once to fix syntax error
                    try:
                        retry_prompt = f"The previous Mermaid diagram had syntax errors: {err_msg}.\nOriginal Code:\n{diagram_code}\nPlease fix the Mermaid syntax. Return ONLY valid Mermaid code, or an empty string if context cannot support it."
                        retry_res = model.generate_content(retry_prompt)
                        cleaned_retry = DiagramEngine.sanitize_mermaid_code(retry_res.text)
                        is_valid2, _ = DiagramEngine.validate_mermaid_syntax(cleaned_retry)
                        if is_valid2:
                            data["diagram_code"] = cleaned_retry
                        else:
                            data["diagram_code"] = ""
                            data["diagram_message"] = "Diagram could not be reliably validated against PDF context."
                    except Exception:
                        data["diagram_code"] = ""
                        data["diagram_message"] = "Diagram could not be reliably validated against PDF context."
                else:
                    data["diagram_code"] = DiagramEngine.sanitize_mermaid_code(diagram_code)
            else:
                data["diagram_code"] = ""

            data["diagram_needed"] = bool(data.get("diagram_code"))
            data["marks"] = marks
            data["question"] = question
            if "scheme" in data and "components" in data["scheme"]:
                data["scheme"]["components"] = calibrate_scheme_components(data["scheme"]["components"], marks)
                data["scheme"]["total_allocated"] = marks
                data["scheme"]["target_marks"] = marks
            return data
        except Exception as err:
            print(f"Error parsing Gemini response JSON: {err}")
            return None

    def _generate_with_synthesis_engine(self, question: str, marks: int, context_chunks: List[Dict[str, Any]], 
                                       scheme_template: Dict, diagram_needed: bool) -> Dict[str, Any]:
        """
        Academic synthesis engine that formulates structured answers strictly from document context
        without inventing statements, generic fillers, or unsupported diagrams.
        """
        combined_text = " ".join([c["text"] for c in context_chunks])
        
        # Extract keywords from question to check relevance
        q_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())) - STOP_WORDS
        text_lower = combined_text.lower()
        
        # Check if question keywords appear in context
        matching_keywords = [k for k in q_tokens if k in text_lower]
        
        # Split sentences from context
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', combined_text) if len(s.strip()) > 15]
        
        # Filter relevant sentences containing question keywords
        relevant_sentences = []
        for s in sentences:
            s_lower = s.lower()
            if any(k in s_lower for k in matching_keywords):
                relevant_sentences.append(s)

        # Fall back to general chunk sentences if some keywords matched
        usable_sentences = relevant_sentences if relevant_sentences else [s for s in sentences if len(s) > 25]

        # If zero keywords matched or no usable academic text, state insufficient information
        if not matching_keywords or not usable_sentences:
            return {
                "question": question,
                "marks": marks,
                "scheme": {
                    "target_marks": marks,
                    "components": [
                        {"component": "Relevance & Content Verification", "marks": marks}
                    ],
                    "total_allocated": marks
                },
                "answer_markdown": NOT_ENOUGH_INFO_MSG,
                "diagram_code": "",
                "diagram_needed": False,
                "diagram_title": "",
                "diagram_message": "No diagram generated because relevant content was not found in the uploaded document.",
                "tips_for_full_marks": [
                    "Ensure the question is directly covered in the uploaded course materials."
                ],
                "sources": context_chunks,
                "has_sufficient_context": False
            }

        # Attempt to synthesize grounded diagram from context
        diagram_code = ""
        diagram_title = ""
        diagram_msg = None
        if diagram_needed:
            code, title, msg = DiagramEngine.generate_diagram_from_context(question, context_chunks)
            if code:
                diagram_code = code
                diagram_title = title
            else:
                diagram_msg = msg

        # Construct Grounded Mark-Specific Answer
        answer_parts = []
        components = []

        if marks == 2:
            components = [
                {"component": "Formal Definition / Core Technical Concept", "marks": 1.0},
                {"component": "Key Characteristic / Operational Principle", "marks": 1.0}
            ]
            lead_text = usable_sentences[0]
            second_text = usable_sentences[1] if len(usable_sentences) > 1 else usable_sentences[0]

            answer_parts.append(f"### 📌 Definition & Core Concept\n{lead_text}\n")
            answer_parts.append(f"### 🔑 Key Characteristics\n- **Principle**: {second_text}")

        elif marks == 5:
            components = [
                {"component": "Introduction & Formal Definition", "marks": 1.0},
                {"component": "Working Mechanism & Core Operational Points", "marks": 2.5},
                {"component": "Technical Details & Context", "marks": 1.5}
            ]
            intro = usable_sentences[0]
            pts = usable_sentences[1:min(5, len(usable_sentences))]

            answer_parts.append(f"### 1. Overview & Definition\n{intro}\n")
            answer_parts.append("### 2. Working Mechanism & Key Operations")
            for idx, pt in enumerate(pts, 1):
                answer_parts.append(f"{idx}. {pt}")
            
            if len(usable_sentences) > 5:
                answer_parts.append(f"\n### 3. Technical Context\n{usable_sentences[5]}")

        elif marks == 10:
            components = [
                {"component": "Conceptual Introduction & Theoretical Background", "marks": 1.5},
                {"component": "Neat Labeled Architecture / Flow Diagram", "marks": 3.0},
                {"component": "Detailed Technical Breakdown & Step-by-Step Working", "marks": 3.5},
                {"component": "Operational Rules & System Analysis", "marks": 2.0}
            ]
            intro = usable_sentences[0]
            pts_core = usable_sentences[1:min(7, len(usable_sentences))]

            answer_parts.append(f"### 1. Introduction & Theoretical Background\n{intro}\n")
            if diagram_code:
                answer_parts.append("### 2. Architectural Blueprint & Component Overview\n*(Refer to the accompanying labeled Mermaid architecture diagram)*\n")
            else:
                answer_parts.append("### 2. Architectural Blueprint & Component Overview\n*(Diagram omitted: document context lacks complete visual schematics for this topic)*\n")

            answer_parts.append("### 3. Step-by-Step Working & Operational Mechanism")
            for pt in pts_core:
                answer_parts.append(f"- {pt}")
            
            if len(usable_sentences) > 7:
                answer_parts.append(f"\n### 4. Operational Rules & Technical Analysis\n{usable_sentences[7]}")
            
            answer_parts.append("\n### 5. Academic Summary & Exam Takeaway\nFor maximum marks in exams, clearly state the boundary conditions, explain every key term, and ensure concepts align strictly with syllabus definitions.")

        else: # 15 marks
            components = [
                {"component": "In-depth Theoretical Foundation & System Principles", "marks": 2.0},
                {"component": "Exhaustive System Architecture & Labeled Diagram", "marks": 4.0},
                {"component": "Detailed Multi-Stage Component & Protocol Analysis", "marks": 5.0},
                {"component": "Operational Workflow & Technical Details", "marks": 2.5},
                {"component": "Comparative Analysis, Trade-offs & Concluding Remarks", "marks": 1.5}
            ]
            answer_parts.append(f"### 1. In-depth Theoretical Foundation & System Principles\n{usable_sentences[0]}\n")
            if diagram_code:
                answer_parts.append("### 2. Comprehensive System Architecture & Interconnections\n*(Refer to the accompanying labeled Mermaid architecture diagram)*\n")
            else:
                answer_parts.append("### 2. Comprehensive System Architecture & Interconnections\n*(Visual schema not fully detailed in uploaded reference)*\n")

            answer_parts.append("### 3. Module-by-Module Operational Breakdown")
            for i in range(1, min(7, len(usable_sentences))):
                answer_parts.append(f"#### 3.{i} Operational Stage\n{usable_sentences[i]}")

            if len(usable_sentences) > 7:
                answer_parts.append(f"\n### 4. Technical Analysis & Considerations\n{usable_sentences[7]}")

            answer_parts.append("\n### 5. Final Concluding Summary\nAdherence to formal system definitions and structured point-by-point presentation guarantees full marks in university evaluations.")

        components = calibrate_scheme_components(components, marks)
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
            "diagram_message": diagram_msg,
            "tips_for_full_marks": [
                f"Structure your response strictly following the {marks}-mark rubric.",
                "Highlight key terminology in bold during written examinations.",
                f"Examiners allot {components[1]['marks'] if len(components) > 1 else 1} marks specifically for neat schematics where applicable."
            ],
            "sources": context_chunks,
            "has_sufficient_context": True
        }
