import os
import json
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from config import PROJECT_INFO, MARK_SCHEMES, UPLOADS_DIR, GEMINI_MODELS
from database import (
    init_db, save_document_record, list_documents,
    save_generated_answer, list_recent_answers,
    save_evaluation, list_recent_evaluations
)
from pdf_processor import AcademicPDFProcessor
from rag_engine import AcademicRAGEngine
from generator import AcademicAnswerGenerator
from evaluator import StudentAnswerEvaluator
from diagram_engine import DiagramEngine
from blueprint_engine import AcademicBlueprintEngine, compute_doc_fingerprint

# Initialize database
init_db()

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Academic Assistant | SIT Tumakuru",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Theme Styling
st.markdown("""
<style>
    /* Main Layout Aesthetics */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #6366f1 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .main-header h1 {
        color: white !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-header h3 {
        color: #e0f2fe !important;
        font-size: 15px;
        font-weight: 500;
        margin: 0 0 12px 0;
    }
    .team-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .team-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    /* Metric and Rubric Cards */
    .rubric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .score-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    .source-box {
        background: #f1f5f9;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .blueprint-q-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .blueprint-q-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 8px rgba(59,130,246,0.1);
    }
    .tag-source {
        background: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        margin-top: 4px;
    }
    .tag-diagram {
        background: #fef3c7;
        color: #b45309;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        margin-top: 4px;
        margin-left: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = AcademicRAGEngine()
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "current_evaluation" not in st.session_state:
    st.session_state.current_evaluation = None
if "selected_question_for_eval" not in st.session_state:
    st.session_state.selected_question_for_eval = ""
if "blueprint_data" not in st.session_state:
    st.session_state.blueprint_data = None
if "blueprint_fingerprint" not in st.session_state:
    st.session_state.blueprint_fingerprint = None
if "is_demo_mode" not in st.session_state:
    st.session_state.is_demo_mode = False
if "user_question_input" not in st.session_state:
    st.session_state.user_question_input = ""
if "target_marks_input" not in st.session_state:
    st.session_state.target_marks_input = 10

sample_pdf_path = UPLOADS_DIR / "Sample_Operating_Systems_Notes.pdf"

# Helper function to load demo mode
def load_sample_demo_mode():
    if sample_pdf_path.exists():
        processor = AcademicPDFProcessor()
        doc_info = processor.process_file(str(sample_pdf_path), "Sample_Operating_Systems_Notes.pdf")
        st.session_state.all_chunks = doc_info["chunks"]
        st.session_state.doc_metadata = [doc_info]
        st.session_state.is_demo_mode = True
        st.session_state.rag_engine.build_index(st.session_state.all_chunks)
        
        # Build blueprint for demo
        bp_engine = AcademicBlueprintEngine(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name="gemini-2.5-flash"
        )
        st.session_state.blueprint_data = bp_engine.generate_blueprint(
            st.session_state.doc_metadata,
            st.session_state.all_chunks
        )
        st.session_state.blueprint_fingerprint = compute_doc_fingerprint(st.session_state.doc_metadata)

# On initial first load, load sample demo if nothing exists
if not st.session_state.all_chunks and "initial_loaded" not in st.session_state:
    st.session_state.initial_loaded = True
    load_sample_demo_mode()

# Header Banner
st.markdown(f"""
<div class="main-header">
    <h1>🏛️ {PROJECT_INFO['institution']}</h1>
    <h3>{PROJECT_INFO['department']} • Mini Project ({PROJECT_INFO['academic_year']})</h3>
    <div style="font-size: 18px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">
        📖 {PROJECT_INFO['title']}
    </div>
    <div class="team-badge-container">
        {"".join([f'<div class="team-badge">👤 {m["name"]} ({m["usn"]})</div>' for m in PROJECT_INFO['team']])}
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Academic Control Panel")
    
    # LLM Settings
    st.subheader("🔑 AI Model Configuration")
    user_api_key = st.text_input(
        "Gemini API Key (Optional)",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Leave blank to run in Autonomous Academic Synthesis mode (offline ready)."
    )
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        st.session_state.rag_engine.api_key = user_api_key
    
    selected_model = st.selectbox(
        "Model Tier",
        options=GEMINI_MODELS,
        index=0
    )
    
    st.divider()

    # Document Uploader
    st.subheader("📚 Academic Document Repository")
    uploaded_files = st.file_uploader(
        "Upload Course Notes / Question Bank (PDF)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    replace_existing = st.checkbox(
        "Replace existing documents with new upload",
        value=True,
        help="When checked, previous notes and demo data are cleared so questions are generated strictly from the new upload."
    )

    if uploaded_files:
        if st.button("📥 Process & Index Uploaded Documents", use_container_width=True, type="primary"):
            processor = AcademicPDFProcessor()
            new_chunks = []
            new_metadata = []
            
            # If replacing, clear old documents and cache
            if replace_existing:
                st.session_state.all_chunks = []
                st.session_state.doc_metadata = []
                st.session_state.is_demo_mode = False
                st.session_state.blueprint_data = None
                st.session_state.blueprint_fingerprint = None
                st.session_state.current_answer = None
                st.session_state.current_evaluation = None
                st.session_state.user_question_input = ""
            
            with st.spinner("Processing documents, extracting academic text, and chunking..."):
                for f in uploaded_files:
                    save_path = UPLOADS_DIR / f.name
                    with open(save_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    
                    info = processor.process_file(str(save_path), f.name)
                    save_document_record(
                        filename=f.name,
                        file_path=str(save_path),
                        total_pages=info["total_pages"],
                        chunk_count=info["total_chunks"],
                        file_size=len(f.getbuffer())
                    )
                    new_chunks.extend(info["chunks"])
                    new_metadata.append(info)
                
                st.session_state.all_chunks.extend(new_chunks)
                st.session_state.doc_metadata.extend(new_metadata)
                st.session_state.rag_engine.build_index(st.session_state.all_chunks)
                
                # Immediately generate fresh dynamic blueprint for the new document set
                bp_engine = AcademicBlueprintEngine(
                    api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                    model_name=selected_model
                )
                st.session_state.blueprint_data = bp_engine.generate_blueprint(
                    st.session_state.doc_metadata,
                    st.session_state.all_chunks
                )
                st.session_state.blueprint_fingerprint = compute_doc_fingerprint(st.session_state.doc_metadata)

            st.success(f"Indexed {len(uploaded_files)} file(s) into {len(st.session_state.all_chunks)} chunks! Exam blueprint refreshed.")

    # Indexed Summary
    st.markdown("### 📊 Knowledge Base Stats")
    total_docs = len(st.session_state.doc_metadata)
    total_chunks = len(st.session_state.all_chunks)
    total_pages = sum([d.get("total_pages", 1) for d in st.session_state.doc_metadata])
    
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Indexed Docs", total_docs)
    col_sb2.metric("Total Pages", total_pages)
    st.metric("RAG Chunks", total_chunks)

    # Document management buttons
    st.subheader("🛠️ Document Actions")
    if st.button("🧪 Load Sample OS Notes (Demo Mode)", use_container_width=True):
        load_sample_demo_mode()
        st.rerun()

    if st.button("🗑️ Clear All Indexed Documents", use_container_width=True):
        st.session_state.all_chunks = []
        st.session_state.doc_metadata = []
        st.session_state.is_demo_mode = False
        st.session_state.blueprint_data = None
        st.session_state.blueprint_fingerprint = None
        st.session_state.current_answer = None
        st.session_state.current_evaluation = None
        st.session_state.user_question_input = ""
        st.session_state.rag_engine.build_index([])
        st.rerun()

# Tabs Interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Scheme-Based Q&A Generation",
    "🎯 Student Answer Evaluator",
    "📋 Exam Blueprint & Important Questions",
    "🔍 Document & Chunk Explorer",
    "📜 History & Records"
])

# =========================================================================
# TAB 1: Scheme-Based Answer Generation
# =========================================================================
with tab1:
    st.subheader("📝 Intelligent Question Answering & Evaluation Scheme Generator")
    st.caption("Generates structured answers calibrated strictly according to academic mark rubrics (2M, 5M, 10M, 15M) with transparent point allocation and labeled Mermaid diagrams.")

    # Dynamic Preset Questions derived from currently active blueprint
    active_presets = []
    if st.session_state.blueprint_data and st.session_state.blueprint_data.get("questions_by_marks"):
        for m in [2, 5, 10, 15]:
            for q_item in st.session_state.blueprint_data["questions_by_marks"].get(m, []):
                active_presets.append(q_item["question"])

    selected_preset = st.selectbox(
        "💡 Quick Pick from Active Question Bank:",
        ["-- Select a question or type below --"] + active_presets,
        index=0
    )

    # Determine default question text
    if selected_preset != "-- Select a question or type below --":
        default_q = selected_preset
    elif st.session_state.get("user_question_input"):
        default_q = st.session_state.user_question_input
    else:
        default_q = ""

    user_question = st.text_area(
        "Academic Question:",
        value=default_q,
        height=75,
        placeholder="Enter your academic question here..."
    )

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        default_marks_idx = 2
        preset_target = st.session_state.get("target_marks_input", 10)
        if preset_target in [2, 5, 10, 15]:
            default_marks_idx = [2, 5, 10, 15].index(preset_target)

        target_marks = st.selectbox(
            "Target Marks (SIT/VTU Scheme):",
            options=[2, 5, 10, 15],
            index=default_marks_idx,
            format_func=lambda m: f"{m} Marks — {MARK_SCHEMES[m]['label'].split('(')[1].replace(')', '')}"
        )
    with col_m2:
        scheme_info = MARK_SCHEMES[target_marks]
        st.info(f"**Expected Standard**: {scheme_info['expected_length']} | **Diagram**: {scheme_info['diagram_required']}")

    if st.button("🚀 Generate Mark-Appropriate Answer & Scheme", type="primary", use_container_width=True):
        if not user_question.strip():
            st.warning("Please enter an academic question.")
        elif not st.session_state.all_chunks:
            st.warning("Please upload a course document in the sidebar or load the sample demo notes.")
        else:
            with st.spinner(f"Retrieving semantic chunks and synthesizing {target_marks}-Mark answer..."):
                top_k = 3 if target_marks == 2 else (5 if target_marks == 5 else 7)
                retrieved_chunks = st.session_state.rag_engine.search(user_question, top_k=top_k)
                
                generator = AcademicAnswerGenerator(
                    api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                    model_name=selected_model
                )
                
                result = generator.generate_scheme_and_answer(
                    question=user_question,
                    marks=target_marks,
                    context_chunks=retrieved_chunks
                )
                
                st.session_state.current_answer = result
                st.session_state.selected_question_for_eval = user_question
                
                # Persist to database
                save_generated_answer(
                    doc_id=1 if st.session_state.doc_metadata else None,
                    question=user_question,
                    marks=target_marks,
                    answer_text=result["answer_markdown"],
                    scheme_dict=result["scheme"],
                    diagram_code=result.get("diagram_code", ""),
                    sources=result.get("sources", [])
                )

    # Render Generated Output
    if st.session_state.current_answer:
        ans_data = st.session_state.current_answer
        st.divider()

        # Rubric Card
        st.markdown(f"### 📋 Official University Evaluation Scheme ({ans_data['marks']} Marks)")
        scheme_obj = ans_data.get("scheme", {})
        components_list = scheme_obj.get("components", [])
        
        # Display Rubric Table
        rubric_cols = st.columns([3, 1])
        with rubric_cols[0]:
            for comp in components_list:
                st.markdown(f"- **{comp.get('component')}**: `{comp.get('marks')} Marks`")
        with rubric_cols[1]:
            st.metric("Total Mark Weightage", f"{scheme_obj.get('total_allocated', ans_data['marks'])} M")

        st.divider()

        # Model Answer Card
        st.markdown(f"### ✍️ Model Answer ({ans_data['marks']} Marks Standard)")
        st.markdown(ans_data["answer_markdown"])

        # Labeled Diagram Card
        if ans_data.get("diagram_needed") and ans_data.get("diagram_code"):
            st.divider()
            diag_title = ans_data.get("diagram_title", "Neat Labeled Architecture / Workflow Diagram")
            st.markdown(f"### 📊 {diag_title}")
            
            diagram_html = DiagramEngine.generate_html_embed(ans_data["diagram_code"], diag_title)
            components.html(diagram_html, height=360, scrolling=True)
            
            with st.expander("🔍 View Raw Mermaid Diagram Code"):
                st.code(ans_data["diagram_code"], language="mermaid")

        # Full Mark Tips
        if ans_data.get("tips_for_full_marks"):
            st.divider()
            st.markdown("### 💡 Examiner's Guidelines for Scoring Full Marks")
            for tip in ans_data["tips_for_full_marks"]:
                st.markdown(f"⭐ {tip}")

        # PDF Citations
        if ans_data.get("sources"):
            st.divider()
            st.markdown("### 📑 Verified PDF Source Citations (RAG)")
            for src in ans_data["sources"]:
                st.markdown(f"""
                <div class="source-box">
                    <b>📄 {src.get('source_doc', 'Document')}</b> | <b>Page {src.get('page', 1)}</b> | Relevance Score: <code>{src.get('score', 'High')}</code>
                    <br/><i style="color: #475569;">"{src.get('text', '')[:220]}..."</i>
                </div>
                """, unsafe_allow_html=True)

        # Export Button
        download_content = f"""# {PROJECT_INFO['institution']}
## {PROJECT_INFO['title']}
### Question: {ans_data['question']} ({ans_data['marks']} Marks)

---
### EVALUATION SCHEME
{json.dumps(ans_data.get('scheme', {}), indent=2)}

---
### MODEL ANSWER
{ans_data['answer_markdown']}

---
### DIAGRAM (MERMAID CODE)
```mermaid
{ans_data.get('diagram_code', '')}
```
"""
        st.download_button(
            label="💾 Download Answer Sheet & Scheme (Markdown)",
            data=download_content,
            file_name=f"Answer_{ans_data['marks']}M.md",
            mime="text/markdown"
        )


# =========================================================================
# TAB 2: Student Answer Evaluator
# =========================================================================
with tab2:
    st.subheader("🎯 Scheme-Based Student Answer Evaluator & Grading")
    st.caption("Submit your handwritten/typed answer for automatic rubric-based evaluation. The AI calculates marks, highlights missing concepts, and provides constructive examiner feedback.")

    eval_q = st.text_input(
        "Examination Question:",
        value=st.session_state.selected_question_for_eval or (default_q or "Enter question to evaluate"),
        placeholder="Enter question"
    )
    
    col_ev_m1, col_ev_m2 = st.columns([1, 1])
    with col_ev_m1:
        eval_marks = st.selectbox(
            "Evaluation Mark Base:",
            options=[2, 5, 10, 15],
            index=2,
            key="eval_marks_select"
        )
    with col_ev_m2:
        if st.button("📋 Load Sample Student Answer for Testing"):
            st.session_state.sample_student_input = (
                f"Regarding {eval_q}:\n"
                "This concept provides core functional capability and data management. "
                "It consists of distinct modular components operating across sequential stages. "
                "The primary advantages are modular design and predictable latency."
            )

    student_text = st.text_area(
        "Student Submitted Answer:",
        value=st.session_state.get("sample_student_input", ""),
        height=180,
        placeholder="Write or paste the student's answer here..."
    )

    if st.button("📊 Evaluate Answer Against Marking Scheme", type="primary", use_container_width=True):
        if not student_text.strip():
            st.warning("Please provide a student answer to evaluate.")
        else:
            with st.spinner("Evaluating student response against academic scheme rubrics..."):
                # Retrieve matching scheme
                if st.session_state.current_answer and st.session_state.current_answer.get("question") == eval_q:
                    active_scheme = st.session_state.current_answer["scheme"]
                    model_ans = st.session_state.current_answer["answer_markdown"]
                else:
                    active_scheme = MARK_SCHEMES.get(eval_marks, MARK_SCHEMES[10])
                    model_ans = ""

                evaluator = StudentAnswerEvaluator(
                    api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                    model_name=selected_model
                )
                
                eval_res = evaluator.evaluate_answer(
                    question=eval_q,
                    target_marks=eval_marks,
                    student_answer=student_text,
                    scheme=active_scheme,
                    model_answer=model_ans
                )
                
                st.session_state.current_evaluation = eval_res
                
                # Persist to database
                save_evaluation(
                    answer_id=None,
                    question=eval_q,
                    target_marks=eval_marks,
                    marks_awarded=eval_res["marks_awarded"],
                    student_answer=student_text,
                    feedback=eval_res
                )

    # Render Evaluation Results
    if st.session_state.current_evaluation:
        ev = st.session_state.current_evaluation
        st.divider()

        # Scorecard Banner
        st.markdown(f"""
        <div class="score-card">
            <h2 style="margin: 0; color: #166534; font-size: 32px;">🏆 Score: {ev['marks_awarded']} / {ev['target_marks']} Marks</h2>
            <div style="font-size: 18px; font-weight: 600; color: #15803d; margin-top: 4px;">
                Percentage: {ev['percentage']}% • Grade: {ev['grade']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Rubric Component Breakdown
        st.markdown("### 📊 Scheme Rubric Breakdown")
        comp_evals = ev.get("component_evaluations", [])
        if comp_evals:
            for item in comp_evals:
                col_c1, col_c2, col_c3 = st.columns([3, 1, 1])
                status_color = "#16a34a" if "Full" in item.get("status", "") else ("#eab308" if "Partial" in item.get("status", "") else "#ef4444")
                col_c1.markdown(f"**{item.get('criterion')}**\n<br/><small style='color: #64748b;'>{item.get('comments', '')}</small>", unsafe_allow_html=True)
                col_c2.markdown(f"`{item.get('marks_awarded')} / {item.get('allocated_marks')} M`")
                col_c3.markdown(f"<span style='color: {status_color}; font-weight: 600;'>{item.get('status')}</span>", unsafe_allow_html=True)
                st.divider()

        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            st.markdown("#### ✅ Strengths Observed")
            for s in ev.get("strengths", []):
                st.markdown(f"- {s}")

        with col_fb2:
            st.markdown("#### ⚠️ Missing Points & Concepts")
            for m in ev.get("missing_points", []):
                st.markdown(f"- {m}")

        st.markdown("#### 👨‍🏫 Chief Examiner's Actionable Feedback")
        st.info(ev.get("examiner_feedback", "Focus on structured diagrams and explicit formulas."))


# =========================================================================
# TAB 3: Dynamic Exam Blueprint & Important Questions
# =========================================================================
with tab3:
    st.subheader("📋 Academic Question Bank & Exam Revision Blueprint")
    
    # Check if documents exist
    if not st.session_state.doc_metadata or not st.session_state.all_chunks:
        st.info("ℹ️ No academic documents are currently processed. Please upload course documents in the sidebar (or click 'Load Sample OS Notes (Demo Mode)') to dynamically generate an exam blueprint.")
    else:
        current_fp = compute_doc_fingerprint(st.session_state.doc_metadata)
        source_names = [d.get("source_doc", "Document") for d in st.session_state.doc_metadata]

        # Top Bar: Status and Refresh Button
        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            if st.session_state.get("is_demo_mode", False):
                st.caption(f"🧪 **Active Material**: Sample Operating Systems Notes *(Demo Mode)*. Upload your course syllabus/notes in the sidebar to switch to your subject.")
            else:
                st.caption(f"📚 **Active Materials ({len(source_names)})**: {', '.join(source_names)}")
        with col_hdr2:
            refresh_clicked = st.button("🔄 Refresh Exam Blueprint", use_container_width=True)

        # Check Cache vs Regeneration
        needs_regen = (
            st.session_state.blueprint_data is None
            or st.session_state.blueprint_fingerprint != current_fp
            or refresh_clicked
        )

        if needs_regen:
            with st.spinner("Analyzing document structure and synthesizing dynamic exam blueprint..."):
                bp_engine = AcademicBlueprintEngine(
                    api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                    model_name=selected_model
                )
                blueprint = bp_engine.generate_blueprint(
                    st.session_state.doc_metadata,
                    st.session_state.all_chunks
                )
                st.session_state.blueprint_data = blueprint
                st.session_state.blueprint_fingerprint = current_fp
        else:
            blueprint = st.session_state.blueprint_data

        # Render Blueprint Content
        if blueprint.get("status") == "insufficient_content":
            st.warning(blueprint.get("message", "Insufficient academic content to formulate questions."))
        elif blueprint.get("total_questions", 0) == 0:
            st.warning("No examination questions could be extracted from the uploaded document text.")
        else:
            st.success(f"Generated {blueprint['total_questions']} prioritized university exam questions strictly from active document(s).")
            
            q_by_marks = blueprint.get("questions_by_marks", {})
            
            col_bp1, col_bp2 = st.columns(2)

            with col_bp1:
                st.markdown("#### 🟢 2-Mark Short Answer Questions")
                q2_list = q_by_marks.get(2, [])
                if not q2_list:
                    st.caption("No 2-mark definition questions found.")
                else:
                    for i, q in enumerate(q2_list):
                        with st.container():
                            st.markdown(f"""
                            <div class="blueprint-q-card">
                                <b>{i+1}. {q['question']}</b><br/>
                                <span class="tag-source">📄 {q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"⚡ Answer 2M #{i+1} in Tab 1", key=f"btn_2m_{q['source_doc']}_{i}"):
                                st.session_state.user_question_input = q["question"]
                                st.session_state.target_marks_input = 2
                                st.rerun()

                st.markdown("#### 🟡 5-Mark Conceptual Questions")
                q5_list = q_by_marks.get(5, [])
                if not q5_list:
                    st.caption("No 5-mark conceptual questions found.")
                else:
                    for i, q in enumerate(q5_list):
                        with st.container():
                            diag_tag = '<span class="tag-diagram">📊 Schematic</span>' if q.get("has_diagram_requirement") else ''
                            st.markdown(f"""
                            <div class="blueprint-q-card">
                                <b>{i+1}. {q['question']}</b><br/>
                                <span class="tag-source">📄 {q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                                {diag_tag}
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"⚡ Answer 5M #{i+1} in Tab 1", key=f"btn_5m_{q['source_doc']}_{i}"):
                                st.session_state.user_question_input = q["question"]
                                st.session_state.target_marks_input = 5
                                st.rerun()

            with col_bp2:
                st.markdown("#### 🔴 10-Mark Detailed Technical & Diagram Questions")
                q10_list = q_by_marks.get(10, [])
                if not q10_list:
                    st.caption("No 10-mark architectural questions found.")
                else:
                    for i, q in enumerate(q10_list):
                        with st.container():
                            diag_tag = '<span class="tag-diagram">📊 Labeled Diagram Required</span>' if q.get("has_diagram_requirement") else ''
                            st.markdown(f"""
                            <div class="blueprint-q-card">
                                <b>{i+1}. {q['question']}</b><br/>
                                <span class="tag-source">📄 {q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                                {diag_tag}
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"⚡ Answer 10M #{i+1} in Tab 1", key=f"btn_10m_{q['source_doc']}_{i}"):
                                st.session_state.user_question_input = q["question"]
                                st.session_state.target_marks_input = 10
                                st.rerun()

                st.markdown("#### 🟣 15-Mark Comprehensive Treatise Questions")
                q15_list = q_by_marks.get(15, [])
                if not q15_list:
                    st.caption("No 15-mark essay questions generated (requires multi-page syllabus content).")
                else:
                    for i, q in enumerate(q15_list):
                        with st.container():
                            st.markdown(f"""
                            <div class="blueprint-q-card">
                                <b>{i+1}. {q['question']}</b><br/>
                                <span class="tag-source">📄 {q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                                <span class="tag-diagram">📊 Architecture & Analysis</span>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"⚡ Answer 15M #{i+1} in Tab 1", key=f"btn_15m_{q['source_doc']}_{i}"):
                                st.session_state.user_question_input = q["question"]
                                st.session_state.target_marks_input = 15
                                st.rerun()

            st.divider()
            st.info("💡 **Interactive Tip**: Click '⚡ Answer in Tab 1' next to any question above to load it directly into Tab 1 with its exact mark weightage!")


# =========================================================================
# TAB 4: Document Explorer & RAG Knowledge Base
# =========================================================================
with tab4:
    st.subheader("🔍 PDF Document Explorer & RAG Knowledge Base")
    st.caption("Inspect extracted pages, raw text, and segment chunks used by semantic retrieval.")

    if not st.session_state.doc_metadata:
        st.info("No documents indexed yet.")
    else:
        for doc in st.session_state.doc_metadata:
            with st.expander(f"📄 {doc['source_doc']} ({doc['total_pages']} Pages, {doc['total_chunks']} Chunks)", expanded=True):
                st.write(f"**File Location**: `{doc['file_path']}`")
                
                page_sel = st.selectbox(
                    f"Select Page to Preview ({doc['source_doc']}):",
                    options=[p["page_number"] for p in doc["pages"]],
                    key=f"page_sel_{doc['source_doc']}"
                )
                
                selected_page_text = next((p["text"] for p in doc["pages"] if p["page_number"] == page_sel), "")
                st.text_area("Extracted Page Text:", value=selected_page_text, height=220, disabled=True)

        st.divider()
        st.markdown("### 🔎 Semantic Chunk Search Tester")
        search_query = st.text_input("Enter search query to test RAG similarity:")
        if search_query:
            results = st.session_state.rag_engine.search(search_query, top_k=4)
            for r in results:
                st.markdown(f"""
                <div class="source-box">
                    <b>Rank #{r['rank']}</b> | Score: <code>{r['score']}</code> | Page {r['page']} ({r['source_doc']})
                    <br/>{r['text']}
                </div>
                """, unsafe_allow_html=True)


# =========================================================================
# TAB 5: History & Records
# =========================================================================
with tab5:
    st.subheader("📜 Generated Answers & Student Evaluation Records")
    st.caption("Historical database log of generated answers, marking schemes, and student grading sessions.")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("#### 📝 Recent Generated Answers")
        recent_ans = list_recent_answers(limit=8)
        if not recent_ans:
            st.write("No answers generated yet.")
        else:
            for item in recent_ans:
                with st.expander(f"[{item['marks']}M] {item['question'][:45]}..."):
                    st.write(f"**Date**: {item['created_at']}")
                    st.markdown(item["answer_text"][:250] + "...")
    
    with col_h2:
        st.markdown("#### 🎓 Recent Student Evaluations")
        recent_evs = list_recent_evaluations(limit=8)
        if not recent_evs:
            st.write("No student evaluations recorded yet.")
        else:
            for ev_item in recent_evs:
                with st.expander(f"Score: {ev_item['marks_awarded']}/{ev_item['target_marks']} - {ev_item['question'][:35]}..."):
                    st.write(f"**Student Answer Excerpt**: {ev_item['student_answer'][:120]}...")
                    feedback_obj = ev_item.get("feedback", {})
                    st.write(f"**Grade**: {feedback_obj.get('grade', 'N/A')}")
                    st.write(f"**Examiner Note**: {feedback_obj.get('examiner_feedback', '')}")
