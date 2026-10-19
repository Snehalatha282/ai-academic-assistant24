import os
import re
import json
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class StudentAnswerEvaluator:
    """
    Intelligent Scheme-Based Student Answer Evaluation Engine.
    Evaluates student answers against academic marking rubrics, calculates marks,
    highlights missing concepts, and provides constructive examiner feedback.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def evaluate_answer(self, question: str, target_marks: float, student_answer: str,
                        scheme: Dict[str, Any], model_answer: str = "") -> Dict[str, Any]:
        """
        Evaluates student answer against scheme and model answer.
        """
        student_answer = student_answer.strip()
        if not student_answer:
            return {
                "marks_awarded": 0.0,
                "target_marks": target_marks,
                "percentage": 0.0,
                "grade": "F (Incomplete / Blank)",
                "component_evaluations": [],
                "missing_points": ["No answer provided by the student."],
                "strengths": [],
                "examiner_feedback": "Please write your answer to receive an evaluation."
            }

        # Attempt Gemini evaluation if key provided
        if self.api_key:
            try:
                res = self._evaluate_with_gemini(question, target_marks, student_answer, scheme, model_answer)
                if res:
                    return res
            except Exception as e:
                print(f"Gemini evaluation failed, falling back to NLP rubric engine: {e}")

        # Intelligent NLP Rubric Engine
        return self._evaluate_with_nlp(question, target_marks, student_answer, scheme, model_answer)

    def _evaluate_with_gemini(self, question: str, target_marks: float, student_answer: str,
                              scheme: Dict[str, Any], model_answer: str) -> Optional[Dict[str, Any]]:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        rubric_str = json.dumps(scheme.get("components", []), indent=2)

        prompt = f"""
You are an experienced University Professor and Chief Examiner in Computer Science at Siddaganga Institute of Technology (SIT) / VTU.
Evaluate the following student's written examination answer against the official marking scheme and model answer.

QUESTION: {question}
TARGET MARKS: {target_marks}

OFFICIAL MARKING SCHEME:
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
Identify specific missing concepts, formulas, diagram references, or explanations.

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
            return data
        except Exception as e:
            print(f"Error parsing Gemini evaluation: {e}")
            return None

    def _evaluate_with_nlp(self, question: str, target_marks: float, student_answer: str,
                           scheme: Dict[str, Any], model_answer: str) -> Dict[str, Any]:
        """
        Rule-based and NLP semantic similarity evaluator.
        """
        components = scheme.get("components", [
            {"component": "Core Definition & Concepts", "marks": target_marks * 0.4},
            {"component": "Technical Details & Explanation", "marks": target_marks * 0.4},
            {"component": "Examples / Diagram / Formula", "marks": target_marks * 0.2}
        ])

        student_lower = student_answer.lower()
        word_count = len(student_answer.split())

        component_evals = []
        total_awarded = 0.0
        missing_pts = []
        strengths = []

        # Reference tokens from model answer
        ref_text = model_answer if model_answer else question

        # Calculate semantic cosine similarity
        try:
            vec = TfidfVectorizer(stop_words='english')
            tfidf_mat = vec.fit_transform([student_answer, ref_text])
            similarity = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            similarity = 0.5

        for comp in components:
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
                # Check if student referenced diagram keywords
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

            else: # General explanation / points
                length_adequacy = min(1.0, word_count / (target_marks * 25))
                score_factor = (0.5 * similarity) + (0.5 * length_adequacy)

            awarded = round(max_m * score_factor, 2)
            total_awarded += awarded

            status = "Full Marks" if awarded >= max_m * 0.85 else ("Partial Credit" if awarded > 0 else "Missing")
            component_evals.append({
                "criterion": name,
                "allocated_marks": max_m,
                "marks_awarded": awarded,
                "status": status,
                "comments": f"Awarded {awarded}/{max_m} marks based on semantic coverage."
            })

        # Cap total awarded
        total_awarded = min(target_marks, round(total_awarded, 1))
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
