import os
from pathlib import Path
from pdf_processor import AcademicPDFProcessor
from blueprint_engine import AcademicBlueprintEngine, compute_doc_fingerprint
from config import UPLOADS_DIR

def run_blueprint_tests():
    print("==================================================")
    print("[*] Running Dynamic Exam Blueprint Verification Suite")
    print("==================================================")

    processor = AcademicPDFProcessor()
    engine = AcademicBlueprintEngine()

    os_pdf_path = UPLOADS_DIR / "Sample_Operating_Systems_Notes.pdf"
    cn_pdf_path = UPLOADS_DIR / "Sample_Computer_Networks_Notes.pdf"

    assert os_pdf_path.exists(), f"Missing OS sample at {os_pdf_path}"
    assert cn_pdf_path.exists(), f"Missing CN sample at {cn_pdf_path}"

    # ----------------------------------------------------
    # TEST 1: Empty Document State
    # ----------------------------------------------------
    print("\n[Test 1/6] Testing Empty Document State...")
    bp_empty = engine.generate_blueprint([], [])
    assert bp_empty["status"] == "empty", f"Expected 'empty', got {bp_empty['status']}"
    assert bp_empty["total_questions"] == 0
    assert bp_empty["fingerprint"] == "empty"
    print("[PASS] Empty state handled correctly with 0 questions.")

    # ----------------------------------------------------
    # TEST 2: Insufficient Content State
    # ----------------------------------------------------
    print("\n[Test 2/6] Testing Insufficient Content State...")
    tiny_meta = [{"source_doc": "tiny.pdf", "total_pages": 1, "total_chunks": 1}]
    tiny_chunks = [{"text": "Hello world.", "page": 1, "source_doc": "tiny.pdf"}]
    bp_tiny = engine.generate_blueprint(tiny_meta, tiny_chunks)
    assert bp_tiny["status"] == "insufficient_content", f"Expected 'insufficient_content', got {bp_tiny['status']}"
    assert bp_tiny["total_questions"] == 0
    print("[PASS] Insufficient content document properly detected.")

    # ----------------------------------------------------
    # TEST 3: Operating Systems Document Blueprint
    # ----------------------------------------------------
    print("\n[Test 3/6] Testing Blueprint on Operating Systems Notes...")
    os_info = processor.process_file(str(os_pdf_path), "Sample_Operating_Systems_Notes.pdf")
    os_meta = [os_info]
    os_chunks = os_info["chunks"]
    
    fp_os = compute_doc_fingerprint(os_meta)
    bp_os = engine.generate_blueprint(os_meta, os_chunks)
    assert bp_os["status"] == "success"
    assert bp_os["total_questions"] > 0
    assert bp_os["fingerprint"] == fp_os
    
    os_all_q_text = " ".join([
        q["question"] for m in [2, 5, 10, 15] for q in bp_os["questions_by_marks"].get(m, [])
    ]).lower()
    
    # Must contain OS concepts
    assert any(term in os_all_q_text for term in ["paging", "memory", "deadlock", "process", "critical section", "tlb"]), \
        "OS blueprint did not contain expected OS concepts"
    print(f"[PASS] OS Blueprint successfully generated {bp_os['total_questions']} questions across marks (2M, 5M, 10M, 15M).")

    # ----------------------------------------------------
    # TEST 4: Computer Networks Document (Switching Document)
    # ----------------------------------------------------
    print("\n[Test 4/6] Testing Document Switch to Computer Networks (No Stale OS Questions)...")
    cn_info = processor.process_file(str(cn_pdf_path), "Sample_Computer_Networks_Notes.pdf")
    cn_meta = [cn_info]
    cn_chunks = cn_info["chunks"]
    
    fp_cn = compute_doc_fingerprint(cn_meta)
    assert fp_cn != fp_os, "Fingerprint failed to differentiate between OS and CN documents!"
    
    bp_cn = engine.generate_blueprint(cn_meta, cn_chunks)
    assert bp_cn["status"] == "success"
    assert bp_cn["total_questions"] > 0
    
    cn_all_q_text = " ".join([
        q["question"] for m in [2, 5, 10, 15] for q in bp_cn["questions_by_marks"].get(m, [])
    ]).lower()

    # Must contain CN concepts
    assert any(term in cn_all_q_text for term in ["osi", "network", "tcp", "udp", "sliding window", "crc", "transport"]), \
        "CN blueprint did not contain expected CN concepts"

    # STRICT BUG VERIFICATION: Old OS questions must NOT be present!
    assert "belady" not in cn_all_q_text, "Found stale OS question (Belady) in CN blueprint!"
    assert "paging" not in cn_all_q_text, "Found stale OS question (Paging) in CN blueprint!"
    assert "deadlock" not in cn_all_q_text, "Found stale OS question (Deadlock) in CN blueprint!"
    assert "banker" not in cn_all_q_text, "Found stale OS question (Banker) in CN blueprint!"
    
    for m in [2, 5, 10, 15]:
        for q in bp_cn["questions_by_marks"].get(m, []):
            assert q["source_doc"] == "Sample_Computer_Networks_Notes.pdf", \
                f"Question source doc incorrect: {q['source_doc']}"
    print("[PASS] Switching documents successfully purged old questions and generated 100% CN-specific questions.")

    # ----------------------------------------------------
    # TEST 5: Multi-Document Ingestion & Source Tagging
    # ----------------------------------------------------
    print("\n[Test 5/6] Testing Multi-Document Ingestion (OS + CN Combined)...")
    multi_meta = [os_info, cn_info]
    multi_chunks = os_chunks + cn_chunks
    
    fp_multi = compute_doc_fingerprint(multi_meta)
    assert fp_multi != fp_os and fp_multi != fp_cn, "Multi-document fingerprint failed uniqueness test"
    
    bp_multi = engine.generate_blueprint(multi_meta, multi_chunks)
    assert bp_multi["status"] == "success"
    assert bp_multi["total_questions"] > 0
    
    # Check that both source documents are represented
    sources_in_blueprint = set()
    for m in [2, 5, 10, 15]:
        for q in bp_multi["questions_by_marks"].get(m, []):
            sources_in_blueprint.add(q["source_doc"])
            assert "source_doc" in q and q["source_doc"] in ["Sample_Operating_Systems_Notes.pdf", "Sample_Computer_Networks_Notes.pdf"]
            assert "page_reference" in q and isinstance(q["page_reference"], int)

    assert "Sample_Operating_Systems_Notes.pdf" in sources_in_blueprint, "Missing OS questions in multi-doc blueprint"
    assert "Sample_Computer_Networks_Notes.pdf" in sources_in_blueprint, "Missing CN questions in multi-doc blueprint"
    print(f"[PASS] Multi-document blueprint correctly generated questions from both documents with precise source tags.")

    # ----------------------------------------------------
    # TEST 6: Fingerprint Caching and Invalidation
    # ----------------------------------------------------
    print("\n[Test 6/6] Testing Fingerprint Caching & Invalidation Logic...")
    fp1 = compute_doc_fingerprint(os_meta)
    fp2 = compute_doc_fingerprint(os_meta)
    fp3 = compute_doc_fingerprint(cn_meta)
    assert fp1 == fp2, "Identical doc sets produced different fingerprints"
    assert fp1 != fp3, "Different doc sets produced identical fingerprints"
    print("[PASS] Deterministic fingerprint caching and cache invalidation verified.")

    print("\n==================================================")
    print("[SUCCESS] ALL 6 DYNAMIC BLUEPRINT TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_blueprint_tests()
