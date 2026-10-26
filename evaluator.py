import os
import re
import json
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import MARK_SCHEMES, calibrate_scheme_components

class StudentAnswerEvaluator:
    """
    Intelligent Scheme-Based Student Answer Evaluation Engine.
    Evaluates student answers against academic marking rubrics, calculates marks,
    highlights missing concepts, and provides constructive examiner feedback.
    Strictly divides and calibrates rubric marks with respect to target marks.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def evaluate_answer(self, question: str, target_marks: float, student_answer: str,
                        scheme: Optional[Dict[str, Any]], model_answer: str = "") -> Dict[str, Any]:
        """
        Evaluates student answer against scheme and model answer.
        Guarantees that component allocated marks strictly divide with respect to target_marks.
        """
        target_marks = float(target_marks)
        student_answer = student_answer.strip()

        # Resolve and calibrate scheme components
        raw_components = None
        if scheme:
            raw_components = scheme.get("components") or scheme.get("breakdown")
        if not raw_components:
            base_scheme = MARK_SCHEMES.get(int(target_marks), MARK_SCHEMES[10])
            raw_components = base_scheme.get("components") or base_scheme.get("breakdown")

        calibrated_components = calibrate_scheme_components(raw_components, target_marks)

        if not student_answer:
            comp_evals = [
                {
                    "criterion": c.get("component", "Concept"),
                    "allocated_marks": float(c.get("marks", 1.0)),
                    "marks_awarded": 0.0,
                    "status": "Missing",
                    "comments": "No response provided for this rubric component."
                }
                for c in calibrated_components
            ]
            return {
                "marks_awarded": 0.0,
                "target_marks": target_marks,
                "percentage": 0.0,
                "grade": "F (Incomplete / Blank)",
                "component_evaluations": comp_evals,
                "missing_points": ["No answer provided by the student."],
                "strengths": [],
                "examiner_feedback": "Please write your answer to receive an evaluation."
            }

        # Attempt Gemini evaluation if key provided
        if self.api_key:
            try:
                res = self._evaluate_with_gemini(question, target_marks, student_answer, calibrated_components, model_answer)
                if res:
                    return res
            except Exception as e:
                print(f"Gemini evaluation failed, falling back to NLP rubric engine: {e}")

        # Intelligent NLP Rubric Engine
        return self._evaluate_with_nlp(question, target_marks, student_answer, calibrated_components, model_answer)

    def _evaluate_with_gemini(self, question: str, target_marks: float, student_answer: str,
                              calibrated_components: List[Dict[str, Any]], model_answer: str) -> Optional[Dict[str, Any]]:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        rubric_str = json.dumps(calibrated_components, indent=2)

        prompt = f"""
You are an experienced University Professor and Chief Examiner in Computer Science at Siddaganga Institute of Technology (SIT) / VTU.
Evaluate the following student's written examination answer against the official marking scheme and model answer.

QUESTION: {question}
TARGET MARKS: {target_marks}

OFFICIAL MARKING SCHEME (Each component must be scored out of its allocated marks):
{rubric_str}

REFERENCE MODEL ANSWER:
\"\"\"
{model_answer}
\"\"\"

STUDENT'S SUBMITTED ANSWER:
\"\"\"
{student_answer}
\"\"\"

Carefully assess each component of the scheme. Award realistic academic marks with justification.
The sum of allocated_marks across all component_evaluations must equal exactly {target_marks}.
The sum of marks_awarded must equal the overall marks_awarded.

Return ONLY a JSON object with this exact structure:
{{
    "marks_awarded": 7.5,
    "target_marks": {target_marks},
    "percentage": 75.0,
    "grade": "A (Very Good)",
    "component_evaluations": [
        {{
            "criterion": "Name of the component from scheme",
            "allocated_marks": 2.0,
            "marks_awarded": 1.5,
            "status": "Partial Credit",
            "comments": "Good definition, but missed the standard equation."
        }}
    ],
    "strengths": [
        "Clearly defined the fundamental concept",
        "Neat structure and relevant examples"
    ],
    "missing_points": [
        "Did not mention time/space complexity",
        "Omitted diagram component explanation"
    ],
    "examiner_feedback": "Comprehensive advice on how to convert this into a 100% full-mark answer."
}}
"""
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            data = json.loads(response.text)
            raw_evals = data.get("component_evaluations", [])
            calibrated_evals = []

            for i, target_c in enumerate(calibrated_components):
                target_alloc = float(target_c.get("marks", 1.0))
                if i < len(raw_evals):
                    ev_item = raw_evals[i]
                    crit_name = ev_item.get("criterion", target_c.get("component", f"Component {i+1}"))
                    awarded_raw = float(ev_item.get("marks_awarded", 0.0))
                    gemini_alloc = float(ev_item.get("allocated_marks", target_alloc))
                    if gemini_alloc > 0:
                        awarded_norm = round((awarded_raw / gemini_alloc) * target_alloc, 1)
                    else:
                        awarded_norm = min(target_alloc, awarded_raw)
                    awarded_norm = min(target_alloc, max(0.0, awarded_norm))
                    status = ev_item.get("status", "Partial Credit")
                    comments = ev_item.get("comments", f"Awarded {awarded_norm}/{target_alloc} marks.")
                else:
                    crit_name = target_c.get("component", f"Component {i+1}")
                    awarded_norm = round(target_alloc * 0.5, 1)
                    status = "Partial Credit"
                    comments = f"Awarded {awarded_norm}/{target_alloc} marks."

                calibrated_evals.append({
                    "criterion": crit_name,
                    "allocated_marks": target_alloc,
                    "marks_awarded": awarded_norm,
                    "status": status,
                    "comments": comments
                })

            total_awarded = round(sum(c["marks_awarded"] for c in calibrated_evals), 1)
            total_awarded = min(target_marks, total_awarded)

            data["component_evaluations"] = calibrated_evals
            data["target_marks"] = target_marks
            data["marks_awarded"] = total_awarded
            data["percentage"] = round((total_awarded / target_marks) * 100, 1)
            return data
        except Exception as e:
            print(f"Error parsing Gemini evaluation: {e}")
            return None

    def _evaluate_with_nlp(self, question: str, target_marks: float, student_answer: str,
                           calibrated_components: List[Dict[str, Any]], model_answer: str) -> Dict[str, Any]:
        """
        Rule-based and NLP semantic similarity evaluator.
        Calibrates marks strictly to target_marks.
        """
        student_lower = student_answer.lower()
        word_count = len(student_answer.split())

        component_evals = []
        missing_pts = []
        strengths = []

        ref_text = model_answer if model_answer else question

        # Calculate semantic cosine similarity
        try:
            vec = TfidfVectorizer(stop_words='english')
            tfidf_mat = vec.fit_transform([student_answer, ref_text])
            similarity = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            similarity = 0.5

        for comp in calibrated_components:
            name = comp.get("component", "Concept")
            max_m = float(comp.get("marks", 1.0))
            name_lower = name.lower()

            # Check coverage based on component type and text length
            score_factor = 0.0

            if "definition" in name_lower or "concept" in name_lower or "introduction" in name_lower:
                if len(student_answer) > 40:
                    score_factor = min(1.0, 0.6 + 0.5 * similarity)
                    strengths.append(f"Successfully introduced {name.lower()}.")
                else:
                    score_factor = 0.4
                    missing_pts.append(f"Incomplete definition or conceptual foundation for {name}.")

            elif "diagram" in name_lower or "architecture" in name_lower or "schematic" in name_lower:
                diagram_keywords = ["diagram", "figure", "box", "layer", "module", "flow", "arrow", "shown below", "graph"]
                if any(kw in student_lower for kw in diagram_keywords):
                    score_factor = 0.8
                    strengths.append("Addressed architectural/diagrammatic aspects.")
                else:
                    score_factor = 0.2
                    missing_pts.append("Missing labeled schematic/architecture diagram or diagrammatic description.")

            elif "algorithm" in name_lower or "complexity" in name_lower or "formula" in name_lower:
                math_keywords = ["o(", "o(n", "o(1", "time complexity", "formula", "equation", "algorithm", "step 1", "sum"]
                if any(kw in student_lower for kw in math_keywords):
                    score_factor = 0.9
                    strengths.append("Included relevant algorithm/formula/complexity notations.")
                else:
                    score_factor = 0.3
                    missing_pts.append(f"Did not include mathematical notation or algorithmic steps required for {name}.")

            else:
                length_adequacy = min(1.0, word_count / max(1.0, target_marks * 25))
                score_factor = (0.5 * similarity) + (0.5 * length_adequacy)

            # Round awarded marks cleanly to 1 decimal place
            awarded = round(max_m * score_factor, 1)
            awarded = min(max_m, max(0.0, awarded))

            status = "Full Marks" if awarded >= max_m * 0.85 else ("Partial Credit" if awarded > 0 else "Missing")
            component_evals.append({
                "criterion": name,
                "allocated_marks": max_m,
                "marks_awarded": awarded,
                "status": status,
                "comments": f"Awarded {awarded}/{max_m} marks based on semantic coverage."
            })

        total_awarded = round(sum(item["marks_awarded"] for item in component_evals), 1)
        total_awarded = min(target_marks, total_awarded)

        # Ensure sum of marks_awarded equals total_awarded
        diff = round(total_awarded - sum(item["marks_awarded"] for item in component_evals), 1)
        if diff != 0 and component_evals:
            max_item = max(component_evals, key=lambda c: c["allocated_marks"])
            new_val = round(max_item["marks_awarded"] + diff, 1)
            max_item["marks_awarded"] = max(0.0, min(max_item["allocated_marks"], new_val))
            total_awarded = round(sum(item["marks_awarded"] for item in component_evals), 1)

        percentage = round((total_awarded / target_marks) * 100, 1)

        # Grade determination
        if percentage >= 90:
            grade = "O (Outstanding)"
        elif percentage >= 80:
            grade = "A+ (Excellent)"
        elif percentage >= 70:
            grade = "A (Very Good)"
        elif percentage >= 60:
            grade = "B+ (Good)"
        elif percentage >= 50:
            grade = "B (Above Average)"
        elif percentage >= 40:
            grade = "C (Pass)"
        else:
            grade = "F (Re-attempt Recommended)"

        if not missing_pts:
            missing_pts.append("Consider adding more real-world technical use-cases for extra credit.")

        feedback = (
            f"You scored {total_awarded}/{target_marks} ({percentage}%). "
            f"To achieve full marks, focus on resolving the missing components highlighted above, "
            f"particularly ensuring neat labeled diagrams and explicit formulas/algorithms are clearly written."
        )

        return {
            "marks_awarded": total_awarded,
            "target_marks": target_marks,
            "percentage": percentage,
            "grade": grade,
            "component_evaluations": component_evals,
            "strengths": strengths if strengths else ["Basic conceptual understanding displayed."],
            "missing_points": missing_pts,
            "examiner_feedback": feedback
        }
