import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, cast, TypedDict
import numpy as np
import torch

# Set CPU threads to 10 for maximum encoding throughput
torch.set_num_threads(10)

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from rank_bm25 import BM25Okapi
try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, OSError):
    faiss = None
    FAISS_AVAILABLE = False

from app.database.database import SessionLocal
from app.models.document import Document
from app.services.embedding_service import model, generate_embedding
from app.services.reranker_service import reranker_service
from benchmarks.metrics import (
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank_at_k
)


class ChunkConfig(TypedDict):
    id: str
    name: str
    type: str
    size: int
    overlap: int


class ChunkItem(TypedDict):
    id: int
    doc_id: int
    doc_name: str
    chunk_index: int
    content: str


# ----------------------------------------------------------------------
# 1. Chunking Functions
# ----------------------------------------------------------------------

def chunk_text_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pure character-based sliding window chunking."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += step
    return chunks


def chunk_text_sentence_aware(text: str, target_size: int = 1100, overlap: int = 150) -> list[str]:
    """
    Deterministic sentence/paragraph-aware chunking.
    Splits text on paragraph and sentence boundaries.
    Packs sentences until target_size is reached, preserving sentence integrity.
    Maintains approximate overlap by retaining trailing sentences for next chunk.
    """
    if not text.strip():
        return []
    
    raw_sentences = re.split(r'(?<=[.!?\n])\s+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    chunks = []
    current_sentences = []
    current_len = 0
    
    for s in sentences:
        s_len = len(s)
        if current_len + s_len > target_size and current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks.append(chunk_str)
            
            overlap_sentences = []
            overlap_len = 0
            for prev_s in reversed(current_sentences):
                if overlap_len + len(prev_s) <= overlap:
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += len(prev_s)
                else:
                    break
            current_sentences = overlap_sentences
            current_len = overlap_len
            
        current_sentences.append(s)
        current_len += s_len
        
    if current_sentences:
        chunks.append(" ".join(current_sentences))
        
    return chunks


# ----------------------------------------------------------------------
# 2. Text Normalization & Tokenization
# ----------------------------------------------------------------------

def normalize_text(t: str) -> str:
    """Collapses whitespace and lowercases text for robust substring matching."""
    return re.sub(r"\s+", " ", t.lower()).strip()


def tokenize_regex(text: str) -> list[str]:
    """Clean production regex tokenizer."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def weighted_rrf(result_lists: list[tuple[list[int], float]], k: int = 60) -> list[int]:
    """RRF fusion identical to production."""
    scores: dict[int, float] = {}
    for results, weight in result_lists:
        for rank, cid in enumerate(results, start=1):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ----------------------------------------------------------------------
# 3. Isolated In-Memory Search Engine
# ----------------------------------------------------------------------

class IsolatedSearchEngine:
    """Isolated in-memory search index for an experimental chunk set."""
    def __init__(self, chunks: list[ChunkItem] | list[dict[str, Any]], embeddings: np.ndarray):
        self.chunks = chunks
        self.chunk_map: dict[int, str] = {c["id"]: c["content"] for c in chunks}
        self.chunk_ids: list[int] = [c["id"] for c in chunks]
        self.dimension = embeddings.shape[1]
        
        # FAISS Index
        if FAISS_AVAILABLE and faiss is not None:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype("float32"))
        else:
            self.index = None
            self.vectors = embeddings.astype("float32")
            
        # BM25 Index
        corpus_tokens = [tokenize_regex(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search_faiss(self, query_vector: Any, top_k: int = 50) -> list[int]:
        q_vec = np.array(query_vector, dtype="float32").reshape(1, -1)
        if self.index is not None:
            _, indices = self.index.search(q_vec, min(top_k, len(self.chunk_ids)))
            return [self.chunk_ids[i] for i in indices[0] if i != -1]
        else:
            diff = self.vectors - q_vec
            sq_dist = np.sum(diff * diff, axis=1)
            k = min(top_k, len(self.chunk_ids))
            top_indices = np.argsort(sq_dist)[:k]
            return [self.chunk_ids[i] for i in top_indices]

    def search_bm25(self, query: str, top_k: int = 50) -> list[int]:
        tokens = tokenize_regex(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunk_ids[i] for i in top_indices]

    def retrieve(self, query: str, query_emb: Any, rerank_model, top_k: int = 5, rerank_depth: int = 10) -> tuple[list[int], list[int], float]:
        t0 = time.perf_counter()
        f_ids = self.search_faiss(query_emb, top_k=50)
        b_ids = self.search_bm25(query, top_k=50)
        
        # RRF with 1.0 / 1.0 weights
        candidate_ids = weighted_rrf([(f_ids, 1.0), (b_ids, 1.0)], k=60)
        
        # Candidate pool before reranking
        pool_ids = candidate_ids[:rerank_depth]
        
        # Cross-encoder rerank
        pairs = [(query, self.chunk_map[cid]) for cid in pool_ids if cid in self.chunk_map]
        if pairs:
            scores = rerank_model.predict(pairs)
            ranked = sorted(zip(pool_ids, scores), key=lambda x: x[1], reverse=True)
            final_top_k = [cid for cid, _ in ranked[:top_k]]
        else:
            final_top_k = pool_ids[:top_k]
            
        t1 = time.perf_counter()
        return final_top_k, pool_ids, (t1 - t0) * 1000.0


# ----------------------------------------------------------------------
# 4. Main Benchmark Execution
# ----------------------------------------------------------------------

def run_ablation():
    print("=" * 85)
    print("STARTING CONTROLLED CHUNKING ABLATION EXPERIMENT")
    print("=" * 85)

    # Base experiment cache directory
    exp_dir = Path(__file__).resolve().parent / "experiments" / "chunking"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load source document text from SQLite
    db = SessionLocal()
    docs = db.query(Document).filter(Document.workspace_id == 2).order_by(Document.id).all()
    source_texts: dict[int, tuple[str, str]] = {
        cast(int, d.id): (str(d.name), str(d.extracted_text or "")) for d in docs
    }
    db.close()

    # Load 50 questions
    questions_file = Path(__file__).resolve().parent / "benchmark_questions_labeled.json"
    raw_questions = json.load(open(questions_file, encoding="utf-8"))[:50]
    
    # Load verified evidence signatures
    signatures_file = Path(__file__).resolve().parent / "evidence_signatures.json"
    signatures_data = json.load(open(signatures_file, encoding="utf-8"))
    timeline_sigs = signatures_data["timeline"]
    narrative_sigs = signatures_data["narrative"]

    # Pre-load models
    print("\nLoading models...")
    rerank_model = reranker_service._load_model()
    _ = generate_embedding("warmup query")
    print("Models preloaded successfully.")

    # Configurations to test
    configs: list[ChunkConfig] = [
        {"id": "A", "name": "Current Baseline (A)", "type": "fixed", "size": 4000, "overlap": 500},
        {"id": "B", "name": "Configuration B", "type": "fixed", "size": 2000, "overlap": 300},
        {"id": "C", "name": "Configuration C", "type": "fixed", "size": 1500, "overlap": 200},
        {"id": "D", "name": "Configuration D", "type": "fixed", "size": 1200, "overlap": 150},
        {"id": "E", "name": "Configuration E", "type": "fixed", "size": 1000, "overlap": 150},
        {"id": "F", "name": "Configuration F", "type": "fixed", "size": 600, "overlap": 100},
        {"id": "G", "name": "Configuration G", "type": "fixed", "size": 400, "overlap": 80},
        {"id": "S", "name": "Sentence-Aware (S)", "type": "sentence", "size": 1100, "overlap": 150},
    ]

    all_results = {}

    for cfg in configs:
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        cfg_cache = exp_dir / f"config_{cfg_id}"
        cfg_cache.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 85)
        print(f"PROCESSING: {cfg_name} (Size={cfg['size']}, Overlap={cfg['overlap']}, Type={cfg['type']})")
        print("#" * 85)

        chunks_cache_file = cfg_cache / "chunks.json"
        embs_cache_file = cfg_cache / "embeddings.npy"

        # 1. Generate or load chunks
        chunk_list: list[ChunkItem]
        if chunks_cache_file.exists():
            print("  Loading chunks from cache...")
            with open(chunks_cache_file, "r", encoding="utf-8") as f:
                chunk_list = cast(list[ChunkItem], json.load(f))
        else:
            print("  Generating fresh chunks...")
            chunk_list = []
            chunk_id_counter = 1
            for doc_id, (doc_name, raw_doc_text) in source_texts.items():
                doc_text: str = str(raw_doc_text or "")
                if cfg["type"] == "fixed":
                    raw_chunks = chunk_text_fixed(doc_text, chunk_size=cfg["size"], overlap=cfg["overlap"])
                else:
                    raw_chunks = chunk_text_sentence_aware(doc_text, target_size=cfg["size"], overlap=cfg["overlap"])
                    
                for idx, c_text in enumerate(raw_chunks):
                    chunk_list.append({
                        "id": chunk_id_counter,
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "chunk_index": idx,
                        "content": c_text
                    })
                    chunk_id_counter += 1
            with open(chunks_cache_file, "w", encoding="utf-8") as f:
                json.dump(chunk_list, f, indent=2)

        total_chunks = len(chunk_list)
        char_lens = [len(c["content"]) for c in chunk_list]
        avg_chars = float(np.mean(char_lens))
        med_chars = float(np.median(char_lens))

        # Token statistics using model.tokenizer
        token_lens = [len(model.tokenizer.encode(c["content"], truncation=False)) for c in chunk_list]
        avg_tokens = float(np.mean(token_lens))
        med_tokens = float(np.median(token_lens))
        min_tokens = int(np.min(token_lens))
        max_tokens = int(np.max(token_lens))
        
        u128 = sum(1 for t in token_lens if t < 128)
        u192 = sum(1 for t in token_lens if t < 192)
        u256 = sum(1 for t in token_lens if t < 256)
        over_384 = sum(1 for t in token_lens if t > 384)
        
        pct_u128 = (u128 / total_chunks) * 100.0
        pct_u192 = (u192 / total_chunks) * 100.0
        pct_u256 = (u256 / total_chunks) * 100.0
        pct_over_384 = (over_384 / total_chunks) * 100.0

        print(f"  Total Chunks: {total_chunks}")
        print(f"  Character Lengths: Avg = {avg_chars:.1f}, Median = {med_chars:.1f}")
        print(f"  Token Lengths: Avg = {avg_tokens:.1f}, Med = {med_tokens:.1f}, Min = {min_tokens}, Max = {max_tokens}")
        print(f"    <128: {pct_u128:.1f}%, <192: {pct_u192:.1f}%, <256: {pct_u256:.1f}%, >384: {pct_over_384:.1f}%")

        # 2. Generate or load embeddings
        if embs_cache_file.exists():
            print("  Loading embeddings from cache...")
            embs_arr = np.load(embs_cache_file)
        else:
            print(f"  Generating embeddings for {total_chunks} chunks (batch_size=64)...")
            t_embed_start = time.perf_counter()
            texts = [c["content"] for c in chunk_list]
            embs_list = model.encode(texts, batch_size=64, show_progress_bar=True)
            embs_arr = np.array(embs_list, dtype="float32")
            np.save(embs_cache_file, embs_arr)
            print(f"  Embeddings generated in {time.perf_counter() - t_embed_start:.2f}s and cached.")

        # 3. Build isolated search engine
        engine = IsolatedSearchEngine(chunk_list, embs_arr)

        # 4. Map Ground Truth using exact normalized evidence matching
        print("  Mapping ground-truth evidence...")
        gt_mapping = {}
        total_remapped_labels = 0

        norm_chunk_contents = {c["id"]: normalize_text(c["content"]) for c in chunk_list}

        for q in raw_questions:
            qid = q["id"]
            matched_cids = set()
            
            # Check timeline signatures
            t_sigs = timeline_sigs.get(str(qid), [])
            for cid, c_norm in norm_chunk_contents.items():
                for sig in t_sigs:
                    if normalize_text(sig) in c_norm:
                        matched_cids.add(cid)
                        break
                        
            # Check narrative signatures
            n_sigs = narrative_sigs.get(str(qid), [])
            for cid, c_norm in norm_chunk_contents.items():
                for sig in n_sigs:
                    if normalize_text(sig) in c_norm:
                        matched_cids.add(cid)
                        break

            # Fallback if boundary cut phrase in half
            if not matched_cids:
                for sig in t_sigs:
                    words = sig.split()
                    if len(words) >= 4:
                        sub_sig = " ".join(words[:4])
                        for cid, c_norm in norm_chunk_contents.items():
                            if normalize_text(sub_sig) in c_norm:
                                matched_cids.add(cid)
                                break
                    if matched_cids:
                        break

            gt_mapping[qid] = sorted(list(matched_cids))
            total_remapped_labels += len(matched_cids)

        gt_out_file = cfg_cache / "ground_truth.json"
        with open(gt_out_file, "w", encoding="utf-8") as f:
            json.dump(gt_mapping, f, indent=2)

        print(f"  Ground-truth mapped: {len(gt_mapping)} questions, {total_remapped_labels} total chunk assignments.")

        # 5. Run benchmark across 50 questions
        print("  Running retrieval benchmark across 50 questions...")
        hits5 = []
        recalls5 = []
        mrrs5 = []
        candidate_pool_hits = []
        candidate_pool_recalls = []
        latencies = []
        retrieved_chunk_counts = 0
        total_rel_chunks = 0
        zero_hit_qids = []

        q_traces = {}

        # Warm-up engine with 1 query
        warm_emb = generate_embedding("space exploration warmup")
        _ = engine.retrieve("space exploration warmup", warm_emb, rerank_model)

        for q in raw_questions:
            qid = q["id"]
            qtext = q["question"]
            rel_cids = gt_mapping[qid]
            total_rel_chunks += len(rel_cids)

            q_emb = generate_embedding(qtext)
            top5, pool_ids, lat_ms = engine.retrieve(
                query=qtext,
                query_emb=q_emb,
                rerank_model=rerank_model,
                top_k=5,
                rerank_depth=10
            )
            latencies.append(lat_ms)

            # Evaluate top 5
            h5 = calculate_hit_at_k(rel_cids, top5, k=5)
            r5 = calculate_recall_at_k(rel_cids, top5, k=5)
            m5 = calculate_reciprocal_rank_at_k(rel_cids, top5, k=5)

            hits5.append(h5)
            recalls5.append(r5)
            mrrs5.append(m5)

            # Candidate pool (depth 10) metrics before rerank
            pool_h = calculate_hit_at_k(rel_cids, pool_ids, k=10)
            pool_r = calculate_recall_at_k(rel_cids, pool_ids, k=10)
            candidate_pool_hits.append(pool_h)
            candidate_pool_recalls.append(pool_r)

            # Count relevant chunks retrieved in top 5
            rel_retrieved = len(set(rel_cids) & set(top5))
            retrieved_chunk_counts += rel_retrieved

            if h5 == 0:
                zero_hit_qids.append(qid)

            q_traces[qid] = {
                "top5": top5,
                "pool10": pool_ids,
                "relevant": rel_cids,
                "hit5": h5,
                "recall5": r5,
                "mrr5": m5
            }

        macro_hit5 = float(np.mean(hits5)) * 100.0
        macro_rec5 = float(np.mean(recalls5)) * 100.0
        macro_mrr5 = float(np.mean(mrrs5))
        pool_macro_rec = float(np.mean(candidate_pool_recalls)) * 100.0
        pool_macro_hit = float(np.mean(candidate_pool_hits)) * 100.0
        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        cfg_summary = {
            "name": cfg_name,
            "type": cfg["type"],
            "size": cfg["size"],
            "overlap": cfg["overlap"],
            "chunks": total_chunks,
            "avg_chars": avg_chars,
            "med_chars": med_chars,
            "avg_tokens": avg_tokens,
            "med_tokens": med_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "pct_u128": pct_u128,
            "pct_u192": pct_u192,
            "pct_u256": pct_u256,
            "pct_over_384": pct_over_384,
            "remapped_labels": total_remapped_labels,
            "hit_at_5": macro_hit5,
            "recall_at_5": macro_rec5,
            "mrr_at_5": macro_mrr5,
            "candidate_pool_hit": pool_macro_hit,
            "candidate_pool_recall": pool_macro_rec,
            "avg_latency": avg_lat,
            "p95_latency": p95_lat,
            "zero_hit_count": len(zero_hit_qids),
            "zero_hit_qids": zero_hit_qids,
            "retrieved_chunks_count": retrieved_chunk_counts,
            "total_rel_chunks": total_rel_chunks,
            "traces": q_traces
        }
        all_results[cfg_name] = cfg_summary

        print(f"\nRESULTS FOR {cfg_name}:")
        print(f"  Hit@5:    {macro_hit5:5.2f}%")
        print(f"  Recall@5: {macro_rec5:5.2f}%")
        print(f"  MRR@5:    {macro_mrr5:.4f}")
        print(f"  Candidate Pool Recall@10: {pool_macro_rec:5.2f}%")
        print(f"  Latency:  Avg = {avg_lat:6.2f} ms | P95 = {p95_lat:6.2f} ms")
        print(f"  Zero-Hit Questions ({len(zero_hit_qids)}): {zero_hit_qids}")

    # Save summary and traces to disk
    out_file = Path(__file__).resolve().parent / "chunking_ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        clean_dump = {
            k: {field: v[field] for field in v if field != "traces"}
            for k, v in all_results.items()
        }
        json.dump(clean_dump, f, indent=2)

    trace_file = Path(__file__).resolve().parent / "chunking_ablation_traces.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump({k: v["traces"] for k, v in all_results.items()}, f, indent=2)

    print("\n" + "=" * 85)
    print("CHUNKING ABLATION COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {out_file}")
    print(f"Traces saved to:  {trace_file}")
    print("=" * 85)


if __name__ == "__main__":
    run_ablation()
