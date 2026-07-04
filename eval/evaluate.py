# eval/evaluate.py — retrieval accuracy benchmark

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retriever import HybridRetriever

BENCHMARK = [
    {
        "question":          "What is CCL?",
        "expected_keywords": ["CCL", "365 days"],
        "expected_source":   "leave_rule_3_extracted.json",
    },
    {
        "question":          "How many spells of CCL can a single female government servant avail?",
        "expected_keywords": ["six spells", "calendar year"],
        "expected_source":   "leave_rule_3_extracted.json",
    },
    {
        "question":          "What is WRIIL?",
        "expected_keywords": ["Work Related Illness", "injury"],
        "expected_source":   "leave_rule_3_extracted.json",
    },
    {
        "question":          "What happens to Special Disability Leave after the 7th CPC?",
        "expected_keywords": ["WRIIL", "replaced"],
        "expected_source":   "leave_rule_3_extracted.json",
    },
    {
        "question":          "What is MACP?",
        "expected_keywords": ["MACP", "Grade Pay"],
        "expected_source":   "recruitment_rule_1.json",
    },
    {
        "question":          "When were the NIT Non-Teaching Recruitment Rules 2019 issued?",
        "expected_keywords": ["2019", "Oversight Committee"],
        "expected_source":   "recruitment_rule_1.json",
    },
    {
        "question":          "How should earned leave be applied?",
        "expected_keywords": ["prior", "leave application"],
        "expected_source":   "leave_rule_extracted.json",
    },
    {
        "question":          "Child care leave for single male parents",
        "expected_keywords": ["single male", "widower", "divorcee"],
        "expected_source":   "leave_rule_3_extracted.json",
    },
]


def run_eval(retriever: HybridRetriever):
    print("=" * 65)
    print("  Retrieval Accuracy Benchmark")
    print("=" * 65)

    hits = 0
    for idx, case in enumerate(BENCHMARK, 1):
        q             = case["question"]
        chunks, _diag = retriever.retrieve(q)

        all_text    = " ".join(c.content for c in chunks).lower()
        all_sources = [c.source for c in chunks]

        kw_hit  = all(kw.lower() in all_text for kw in case["expected_keywords"])
        src_hit = case["expected_source"] in all_sources
        passed  = kw_hit and src_hit
        hits   += int(passed)

        status = "PASS" if passed else "FAIL"
        print(f"\n[{idx}] {status}  {q}")
        if not kw_hit:
            print(f"     Missing keywords : {case['expected_keywords']}")
        if not src_hit:
            print(f"     Expected source  : {case['expected_source']}")
            print(f"     Got sources      : {list(set(all_sources))}")

    total = len(BENCHMARK)
    print(f"\n{'=' * 65}")
    print(f"  Score: {hits}/{total}  ({hits / total * 100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    r = HybridRetriever()
    run_eval(r)
