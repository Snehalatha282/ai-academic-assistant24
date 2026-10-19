import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional

def compute_doc_fingerprint(doc_metadata: List[Dict[str, Any]]) -> str:
    """Computes a deterministic hash fingerprint representing the current document set."""
    if not doc_metadata:
        return "empty"
    items = []
    for d in doc_metadata:
        doc_name = d.get("source_doc", "unknown")
        pages = d.get("total_pages", 0)
        chunks = d.get("total_chunks", 0)
        items.append(f"{doc_name}:{pages}:{chunks}")
    raw_str = ";".join(sorted(items))
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

class AcademicBlueprintEngine:
    """
    Dynamically generates examination blueprints and prioritized questions
    (2-Mark, 5-Mark, 10-Mark, 15-Mark) from currently active academic documents.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def generate_blueprint(self, doc_metadata: List[Dict[str, Any]], 
                           all_chunks: List[Dict[str, Any]], 
                           force_refresh: bool = False) -> Dict[str, Any]:
        """
        Main entry point for generating dynamic blueprint from currently processed documents.
        """
        if not doc_metadata or not all_chunks:
            return {
                "status": "empty",
                "message": "No academic documents are currently processed. Please upload course documents in the sidebar to generate an exam blueprint.",
                "fingerprint": "empty",
                "source_documents": [],
                "questions_by_marks": {2: [], 5: [], 10: [], 15: []},
                "total_questions": 0
            }

        fingerprint = compute_doc_fingerprint(doc_metadata)

        # Check total extracted text across documents
        total_text_length = sum(len(c.get("text", "")) for c in all_chunks)
        if total_text_length < 80:
            return {
                "status": "insufficient_content",
                "message": "The processed document contains insufficient text to synthesize university examination questions.",
                "fingerprint": fingerprint,
                "source_documents": [d.get("source_doc", "Document") for d in doc_metadata],
                "questions_by_marks": {2: [], 5: [], 10: [], 15: []},
                "total_questions": 0
            }

        # Attempt Gemini Generation if API key is provided
        if self.api_key:
            try:
                gemini_res = self._generate_with_gemini(doc_metadata, all_chunks, fingerprint)
                if gemini_res and gemini_res.get("total_questions", 0) > 0:
                    return gemini_res
            except Exception as e:
                print(f"Gemini blueprint generation failed, falling back to autonomous engine: {e}")

        # Deterministic Academic Extraction Engine
        return self._generate_with_academic_extractor(doc_metadata, all_chunks, fingerprint)

    def _generate_with_gemini(self, doc_metadata: List[Dict[str, Any]], 
                              all_chunks: List[Dict[str, Any]], 
                              fingerprint: str) -> Optional[Dict[str, Any]]:
        import google.genai as genai
        client = genai.Client(api_key=self.api_key)

        # Compile summaries and key chunks per document
        docs_summary = []
        for doc in doc_metadata:
            doc_name = doc.get("source_doc", "Document")
            pages_count = doc.get("total_pages", 1)
            # Sample up to 6 chunks for this document
            doc_chunks = [c for c in all_chunks if c.get("source_doc") == doc_name][:8]
            sample_text = "\n---\n".join([f"[Page {c.get('page', 1)}] {c.get('text', '')[:400]}" for c in doc_chunks])
            docs_summary.append(f"DOCUMENT: {doc_name} ({pages_count} Pages)\n{sample_text}")

        prompt = f"""
You are a Senior University Academic Examiner for Computer Science & Engineering.
Analyze the following extracted course materials and synthesize an official Exam Blueprint containing high-probability examination questions strictly derived from the text.

Extracted Course Materials:
\"\"\"
{chr(10).join(docs_summary)}
\"\"\"

Requirements:
1. Synthesize questions strictly based on the extracted content. Do NOT invent unrelated questions.
2. Group questions into 4 mark categories: 2 Marks, 5 Marks, 10 Marks, and 15 Marks.
3. For each question, specify the exact source document name and relevant page number.
4. Ensure 2-mark questions test definitions, notations, and core statements.
5. Ensure 5-mark questions test working mechanisms, components, and comparisons.
6. Ensure 10-mark questions require neat labeled architecture/diagrams and comprehensive step-by-step algorithms.
7. Ensure 15-mark questions require deep multi-module treatise with system architecture and trade-off analysis.

Return ONLY a JSON object with this exact schema:
{{
    "status": "success",
    "summary": "Brief 1-2 sentence academic summary of the indexed syllabus topics",
    "questions": [
        {{
            "question": "Question text here",
            "marks": 2,
            "source_doc": "Exact document name",
            "page_reference": 1,
            "core_concept": "Concept name",
            "has_diagram_requirement": false
        }}
    ]
}}
"""

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        try:
            data = json.loads(response.text)
            raw_questions = data.get("questions", [])
            q_by_marks = {2: [], 5: [], 10: [], 15: []}
            for q in raw_questions:
                m = int(q.get("marks", 5))
                if m in q_by_marks:
                    q_by_marks[m].append(q)
                else:
                    q_by_marks[10].append(q)

            total_q = sum(len(v) for v in q_by_marks.values())
            return {
                "status": "success",
                "message": data.get("summary", "Exam Blueprint generated from active documents."),
                "fingerprint": fingerprint,
                "source_documents": [d.get("source_doc", "Document") for d in doc_metadata],
                "questions_by_marks": q_by_marks,
                "total_questions": total_q
            }
        except Exception as err:
            print(f"Error parsing Gemini blueprint response: {err}")
            return None

    def _generate_with_academic_extractor(self, doc_metadata: List[Dict[str, Any]], 
                                          all_chunks: List[Dict[str, Any]], 
                                          fingerprint: str) -> Dict[str, Any]:
        """
        Deterministic, offline-ready academic question synthesis engine.
        Parses headings, technical definitions, algorithms, and processes directly from extracted text.
        """
        questions_by_marks = {2: [], 5: [], 10: [], 15: []}
        seen_questions = set()

        for doc in doc_metadata:
            doc_name = doc.get("source_doc", "Document")
            doc_chunks = [c for c in all_chunks if c.get("source_doc") == doc_name]

            # Extract concepts, headings, and algorithms from doc_chunks
            extracted_concepts = self._extract_concepts_from_chunks(doc_chunks, doc_name)

            for item in extracted_concepts:
                concept = item["concept"]
                page = item["page"]
                kind = item["kind"]  # 'heading', 'definition', 'algorithm', 'comparison'

                # Formulate 2-Mark Question
                if kind in ['definition', 'heading']:
                    q2_text = f"Define {concept} and state its primary purpose."
                    if q2_text not in seen_questions:
                        seen_questions.add(q2_text)
                        questions_by_marks[2].append({
                            "question": q2_text,
                            "marks": 2,
                            "source_doc": doc_name,
                            "page_reference": page,
                            "core_concept": concept,
                            "has_diagram_requirement": False
                        })

                # Formulate 5-Mark Question
                if kind in ['algorithm', 'process', 'heading', 'comparison']:
                    if kind == 'comparison':
                        q5_text = f"Differentiate between the key techniques discussed in {concept}."
                    elif kind == 'algorithm':
                        q5_text = f"Explain the working steps and data structures of {concept}."
                    else:
                        q5_text = f"Explain the core working mechanism and components of {concept} with an example."

                    if q5_text not in seen_questions:
                        seen_questions.add(q5_text)
                        questions_by_marks[5].append({
                            "question": q5_text,
                            "marks": 5,
                            "source_doc": doc_name,
                            "page_reference": page,
                            "core_concept": concept,
                            "has_diagram_requirement": False
                        })

                # Formulate 10-Mark Question (Requires Architecture / Diagram)
                if kind in ['heading', 'algorithm', 'system']:
                    q10_text = f"Explain {concept} in detail with a neat labeled architecture diagram, operational workflow, and key criteria."
                    if q10_text not in seen_questions:
                        seen_questions.add(q10_text)
                        questions_by_marks[10].append({
                            "question": q10_text,
                            "marks": 10,
                            "source_doc": doc_name,
                            "page_reference": page,
                            "core_concept": concept,
                            "has_diagram_requirement": True
                        })

                # Formulate 15-Mark Question (Comprehensive Essay for Major Topics)
                if kind in ['system', 'heading'] and len(doc_chunks) >= 3:
                    q15_text = f"Provide a comprehensive technical treatise on {concept}: explain its architectural foundations with a labeled diagram, analyze multi-stage execution, and discuss trade-offs and complexity."
                    if q15_text not in seen_questions and len(questions_by_marks[15]) < 3:
                        seen_questions.add(q15_text)
                        questions_by_marks[15].append({
                            "question": q15_text,
                            "marks": 15,
                            "source_doc": doc_name,
                            "page_reference": page,
                            "core_concept": concept,
                            "has_diagram_requirement": True
                        })

        # Ensure we don't overwhelm the UI; limit to top 5-6 per category while keeping diverse topics
        for m in questions_by_marks:
            questions_by_marks[m] = questions_by_marks[m][:6]

        total_q = sum(len(v) for v in questions_by_marks.values())
        source_names = [d.get("source_doc", "Document") for d in doc_metadata]

        return {
            "status": "success",
            "message": f"Successfully synthesized {total_q} exam questions from {len(source_names)} active document(s): {', '.join(source_names)}.",
            "fingerprint": fingerprint,
            "source_documents": source_names,
            "questions_by_marks": questions_by_marks,
            "total_questions": total_q
        }

    def _extract_concepts_from_chunks(self, chunks: List[Dict[str, Any]], doc_name: str) -> List[Dict[str, Any]]:
        """Extracts key concepts, section titles, algorithms, and technical terms from text chunks."""
        extracted = []
        seen = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            page = chunk.get("page", 1)

            # 1. Section Headings (e.g., "1.1 Concept of Virtual Memory", "MODULE 1: ...", "Paging Hardware")
            lines = text.split("\n")
            for line in lines:
                line_str = line.strip()
                if not line_str or len(line_str) > 75:
                    continue

                # Match patterns like "1.2 Paging Hardware and Address Translation", "MODULE 2: ...", "2.1 Process Concept"
                heading_match = re.match(r'^(?:(?:Module|Unit|Chapter)\s+\d+[:\s\-\.]+)?(?:\d+\.[\d\.]*\s+)?([A-Z][\w\s\(\)\/\,\-]+)$', line_str)
                if heading_match:
                    concept_candidate = heading_match.group(1).strip()
                    # Clean common noisy prefixes
                    concept_candidate = re.sub(r'^(?:Concept of|Introduction to|Overview of)\s+', '', concept_candidate, flags=re.IGNORECASE).strip()
                    if len(concept_candidate) >= 4 and concept_candidate.lower() not in seen and not concept_candidate.isdigit():
                        seen.add(concept_candidate.lower())
                        is_algo = bool(re.search(r'algorithm|scheduling|replacement|protocol|sorting', concept_candidate, re.I))
                        is_system = bool(re.search(r'architecture|system|memory|network|database|deadlock', concept_candidate, re.I))
                        extracted.append({
                            "concept": concept_candidate,
                            "page": page,
                            "kind": 'algorithm' if is_algo else ('system' if is_system else 'heading')
                        })

            # 2. Key Defined Terms (e.g. "X is a ...", "X is defined as ...", "X refers to ...")
            def_patterns = [
                r'([A-Z][\w\s]{2,30}?)\s+(?:is\s+(?:a|an|the)|is\s+defined\s+as|refers\s+to)\s+[^.!?]{10,80}',
                r'([A-Z][\w\s]{2,25})\s+Algorithm',
                r'([A-Z][\w\s]{2,25})\s+Protocol'
            ]
            for pattern in def_patterns:
                for match in re.finditer(pattern, text):
                    term = match.group(1).strip()
                    # Clean trailing prepositions or short words
                    term = re.sub(r'\b(?:and|or|in|of|for|with)\b.*$', '', term).strip()
                    if 3 < len(term) < 40 and term.lower() not in seen and not any(term.lower() in s for s in seen):
                        # Filter out common false positives
                        if term.lower() not in ['this', 'that', 'there', 'it', 'these', 'those', 'such', 'what', 'which', 'an', 'the']:
                            seen.add(term.lower())
                            is_algo = "algorithm" in term.lower() or "protocol" in term.lower()
                            extracted.append({
                                "concept": term,
                                "page": page,
                                "kind": 'algorithm' if is_algo else 'definition'
                            })

        # If too few concepts found via regex, extract prominent noun phrases / capitalized phrases
        if len(extracted) < 3 and chunks:
            # Fallback extraction from title or first non-empty lines
            first_text = chunks[0].get("text", "")
            first_lines = [l.strip() for l in first_text.splitlines() if 5 < len(l.strip()) < 60]
            for l in first_lines[:3]:
                if l.lower() not in seen:
                    seen.add(l.lower())
                    extracted.append({
                        "concept": l,
                        "page": chunks[0].get("page", 1),
                        "kind": 'heading'
                    })

        return extracted
