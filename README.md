# AI Academic Assistant: Intelligent PDF Analysis and Scheme-Based Answer Generation

**Department of Computer Science and Engineering**  
**Siddaganga Institute of Technology, Tumakuru – 572103**  
*Academic Year 2025–26*

---

## 👥 Student Team Details

| Name | USN | Mobile |
| :--- | :--- | :--- |
| **Shamitha H S** | `1SI24CS161` | 7975348582 |
| **Skarma Lamo** | `1SI24CS172` | 7006820598 |
| **Snehalatha** | `1SI24CS174` | 8073815036 |
| **Sharanya H M** | `1SI24CS213` | 8310968958 |

---

## 📌 Problem Statement

Students frequently struggle to extract precise, exam-aligned answers from vast academic materials such as lecture notes, reference textbooks, and question papers. Conventional keyword search often fails to capture semantic meaning or context. Furthermore, students struggle to gauge the appropriate depth, structure, formulas, and diagrams required for different mark weightages (**2, 5, 10, or 15 marks**).

This project delivers an **AI Academic Assistant** that:
1. Ingests academic PDFs (lecture slides, notes, syllabi, textbooks).
2. Performs RAG-based semantic chunk retrieval with source page citations.
3. Formulates structured model answers calibrated strictly to **2, 5, 10, or 15 marks**.
4. Produces transparent **Evaluation Schemes (Rubrics)** breaking down expected point-by-point mark distribution.
5. Automatically synthesizes and renders **labeled architecture/workflow diagrams** using **Mermaid.js**.
6. Evaluates a student's submitted answer against the official scheme, computes marks, identifies missing concepts, and provides constructive examiner feedback.

---

## 🚀 Key Features

- **Mark-Calibrated Academic Rubric Engine**:
  - **2 Marks**: Crisp definition, principle, 2 key bullet points or core equation.
  - **5 Marks**: Overview, 3-4 numbered operational points, schematic or syntax, practical example.
  - **10 Marks**: Introduction, neat labeled diagram (3 Marks), working mechanism (3.5 Marks), complexity/formula (2 Marks).
  - **15 Marks**: Comprehensive multi-section breakdown, architectural diagram, algorithmic workflows, comparative table, summary.
- **RAG & Semantic Retrieval**: Hybrid semantic retrieval using TF-IDF n-grams + cosine similarity with support for dense embeddings (Gemini/Sentence-Transformers).
- **Mermaid.js Diagram Synthesis**: Programmatically generates flowcharts, state transitions, layered architectures, and sequence diagrams with sanitized syntax.
- **Student Answer Auto-Grader**: Analyzes student answers, scores each rubric component, calculates total marks and percentage, assigns VTU letter grades (O, A+, A, B+, B, C, F), highlights missing concepts, and gives examiner tips.
- **Auto Question Paper & Revision Blueprint**: Extracts high-probability 2M, 5M, 10M, and 15M questions from uploaded documents.
- **Dual-Mode Operation**:
  - **Online Mode**: Uses Google Gemini API (`gemini-2.5-flash`, `gemini-2.5-pro`).
  - **Offline / Autonomous Mode**: Built-in academic knowledge synthesis engine that runs without requiring an internet connection or API key during lab vivas.

---

## 🛠️ Technologies Used

- **Frontend**: Streamlit with custom CSS and interactive HTML components
- **Backend**: Python 3.11
- **PDF Processing**: PyMuPDF (`fitz`), text sanitization, structural chunking
- **RAG & NLP**: Sentence chunking, TF-IDF n-gram vectorization, Cosine Similarity, Gemini Embeddings
- **AI Answer Generation**: Google Gemini API / Academic Knowledge Synthesis Engine
- **Diagram Generation**: Mermaid.js, SVG rendering
- **Database**: SQLite (`academic_assistant.db`)

---

## 📂 Project Architecture

```
ai-academic-assistant/
├── app.py                      # Main Streamlit web application & UI tabs
├── config.py                   # Academic mark rubrics, institution info, paths
├── pdf_processor.py            # PDF text extraction, metadata, page chunking
├── rag_engine.py               # RAG semantic retrieval & vector search engine
├── generator.py                # Mark-appropriate answer & rubric generation engine
├── blueprint_engine.py         # Dynamic Exam Blueprint & multi-document question synthesizer
├── evaluator.py                # Student answer auto-grader and feedback engine
├── diagram_engine.py           # Programmatic Mermaid diagram synthesizer
├── database.py                 # SQLite database layer for persistence
├── generate_sample_pdf.py      # Creates standard SIT OS course notes PDF
├── generate_cn_sample_pdf.py   # Creates standard SIT Computer Networks notes PDF
├── test_system.py              # Automated 5-stage system test suite
├── test_blueprint_dynamic.py   # Dynamic blueprint verification suite (6 tests)
├── verify_features.py          # End-to-end UI features test suite
├── data/
│   ├── uploads/                # Stored academic PDFs and notes
│   │   ├── Sample_Operating_Systems_Notes.pdf
│   │   └── Sample_Computer_Networks_Notes.pdf
│   └── academic_assistant.db   # SQLite records
└── requirements.txt            # Python dependencies
```

---

## ⚡ Quick Start & Execution

### 1. Launch the Web Application
```powershell
python -m streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 2. Run the System Test Suite
```powershell
python test_system.py
```

### 3. Optional: Configure Gemini API Key
You can enter your API key directly in the sidebar or add it to a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If no API key is provided, the system seamlessly runs in Autonomous Academic Synthesis mode)*.
