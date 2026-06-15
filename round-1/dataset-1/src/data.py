#!/usr/bin/env python3
"""Load 6 logical/symbolic reasoning datasets and standardize to exp_sel_data_out schema."""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data_run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT = WORKSPACE / "full_data_out.json"
MAX_ROWS_LARGE = 5000


def load_json(path: Path) -> list:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f}MB)")
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try JSONL: each non-empty line is a JSON object (may have leading '[' / trailing ']')
        rows = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        logger.info(f"  Parsed as JSONL: {len(rows)} rows")
        return rows


def process_clutrr(rows: list) -> list:
    examples = []
    for r in rows:
        story = r.get("story", "") or r.get("clean_story", "")
        query = r.get("query", "")
        question = f"Story: {story}\nQuery: {query}\nWhat is the family relationship?"
        answer = str(r.get("target_text") or r.get("answer", ""))
        ex = {
            "input": question,
            "output": answer,
            "metadata_source_id": str(r.get("id", "")),
            "metadata_task_name": str(r.get("task_name", "")),
            "metadata_f_comb": str(r.get("f_comb", "")),
            "metadata_proof_state": str(r.get("proof_state", ""))[:500],
            "metadata_hop_depth": str(len(str(r.get("edge_types", "[]")).split(","))),
            "metadata_task_type": "relation_classification",
        }
        examples.append(ex)
    return examples


def process_ruletaker(rows: list) -> list:
    examples = []
    for r in rows:
        context = r.get("context", "")
        question = r.get("question", "")
        inp = f"Context: {context}\nAssertion: {question}\nIs this assertion entailed?"
        label = str(r.get("label", ""))
        config = str(r.get("config", ""))
        depth = "0"
        if "depth-" in config:
            depth = config.split("depth-")[-1]
        ex = {
            "input": inp,
            "output": label,
            "metadata_config": config,
            "metadata_hop_depth": depth,
            "metadata_task_type": "logical_entailment",
        }
        examples.append(ex)
    return examples


def process_folio(rows: list) -> list:
    examples = []
    for r in rows:
        premises = r.get("premises", "")
        premises_fol = r.get("premises-FOL", "")
        conclusion = r.get("conclusion", "")
        conclusion_fol = r.get("conclusion-FOL", "")
        inp = (
            f"Premises:\n{premises}\n\n"
            f"Premises (FOL):\n{premises_fol}\n\n"
            f"Conclusion: {conclusion}\n"
            f"Conclusion (FOL): {conclusion_fol}\n\n"
            f"Is the conclusion True, False, or Unknown given the premises?"
        )
        label = str(r.get("label", ""))
        ex = {
            "input": inp,
            "output": label,
            "metadata_story_id": str(r.get("story_id", "")),
            "metadata_example_id": str(r.get("example_id", "")),
            "metadata_premises_fol": premises_fol[:300],
            "metadata_conclusion_fol": conclusion_fol[:200],
            "metadata_task_type": "fol_entailment",
        }
        examples.append(ex)
    return examples


def process_proofwriter(rows: list) -> list:
    examples = []
    for r in rows:
        theory = r.get("theory", "")
        question = r.get("question", "")
        inp = f"Theory:\n{theory}\n\nQuery: {question}\nIs this True, False, or Unknown?"
        answer = str(r.get("answer", ""))
        ex = {
            "input": inp,
            "output": answer,
            "metadata_id": str(r.get("id", "")),
            "metadata_hop_depth": str(r.get("QDep", "")),
            "metadata_max_depth": str(r.get("maxD", "")),
            "metadata_n_facts": str(r.get("NFact", "")),
            "metadata_n_rules": str(r.get("NRule", "")),
            "metadata_config": str(r.get("config", "")),
            "metadata_proof": str(r.get("allProofs", ""))[:500],
            "metadata_task_type": "logical_proof",
        }
        examples.append(ex)
    return examples


def process_musique(rows: list) -> list:
    examples = []
    for r in rows:
        question = r.get("question", "")
        paragraphs = r.get("paragraphs", [])
        context_parts = []
        for p in paragraphs[:5]:
            if isinstance(p, dict):
                title = p.get("title", "")
                text = p.get("paragraph_text", "")
                context_parts.append(f"[{title}] {text}")
        context = "\n".join(context_parts)
        inp = f"Context:\n{context}\n\nQuestion: {question}"
        answer = str(r.get("answer", ""))
        decomp = r.get("question_decomposition", [])
        n_hops = len(decomp) if decomp else 0
        ex = {
            "input": inp,
            "output": answer,
            "metadata_id": str(r.get("id", "")),
            "metadata_hop_depth": str(n_hops),
            "metadata_answerable": str(r.get("answerable", True)),
            "metadata_task_type": "multihop_qa",
        }
        examples.append(ex)
    return examples


def process_babi(rows: list) -> list:
    examples = []
    for r in rows:
        passage = r.get("passage", "")
        question = r.get("question", "")
        inp = f"Story:\n{passage}\n\nQuestion: {question}"
        answer = str(r.get("answer", ""))
        ex = {
            "input": inp,
            "output": answer,
            "metadata_task": str(r.get("task", "")),
            "metadata_task_type": "story_qa",
        }
        examples.append(ex)
    return examples


def main():
    Path("logs").mkdir(exist_ok=True)

    dataset_configs = [
        ("CLUTRR", "full_kendrivp_CLUTRR_v1_extracted_default_train.json", process_clutrr, MAX_ROWS_LARGE),
        ("RuleTaker", "full_tasksource_ruletaker_default_train.json", process_ruletaker, MAX_ROWS_LARGE),
        ("FOLIO", "full_tasksource_folio_default_train.json", process_folio, None),
        ("ProofWriter", "full_tasksource_proofwriter_default_train.json", process_proofwriter, MAX_ROWS_LARGE),
        ("MuSiQue", "full_dgslibisey_MuSiQue_default_train.json", process_musique, MAX_ROWS_LARGE),
        ("bAbI", "full_Muennighoff_babi_default_train.json", process_babi, None),
    ]

    all_datasets = []
    for name, filename, processor, max_rows in dataset_configs:
        path = DATASETS_DIR / filename
        if not path.exists():
            logger.warning(f"Missing: {path}")
            continue
        try:
            rows = load_json(path)
            if max_rows and len(rows) > max_rows:
                logger.info(f"{name}: capping {len(rows)} → {max_rows} rows")
                rows = rows[:max_rows]
            examples = processor(rows)
            logger.info(f"{name}: {len(examples)} examples processed")
            all_datasets.append({"dataset": name, "examples": examples})
        except Exception:
            logger.error(f"Failed processing {name}")
            raise

    output = {"datasets": all_datasets}
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    total = sum(len(d["examples"]) for d in all_datasets)
    logger.info(f"Saved {len(all_datasets)} datasets, {total} total examples → {OUTPUT}")


if __name__ == "__main__":
    main()
