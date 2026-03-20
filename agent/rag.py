"""RAG retrieval module — semantic search over Abaqus documentation.

Uses ChromaDB for vector storage and sentence-transformers for local embeddings.
Supports two data sources:
1. Abaqus HTML documentation (parsed with BeautifulSoup)
2. Manual JSON documentation (hand-curated API entries)
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Default embedding model — multilingual, best Hit@1 on benchmark (100%)
_EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
_COLLECTION_NAME = "abaqus_docs_v2"
# Distance threshold — ignore results above this (lower = more relevant)
# distiluse model uses higher distances than MiniLM, so threshold is higher
_MAX_DISTANCE = 0.85


class AbaqusRAG:
    """Semantic retrieval over Abaqus API documentation."""

    def __init__(self, db_dir="rag_db"):
        """Initialize ChromaDB persistent storage and embedding model.

        Args:
            db_dir: Directory for ChromaDB persistence. Relative paths are
                resolved from the project root (one level up from agent/).
        """
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        # Resolve db_dir relative to project root
        if not os.path.isabs(db_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_dir = os.path.join(project_root, db_dir)

        self._db_dir = db_dir

        # Use offline mode if model is already cached to avoid network issues
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._ef = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
        self._client = chromadb.PersistentClient(path=db_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
        )
        logger.info("AbaqusRAG initialized (db=%s, docs=%d)", db_dir, self._collection.count())

    # ------------------------------------------------------------------
    # Ingestion: HTML docs
    # ------------------------------------------------------------------

    def ingest_html_docs(self, docs_dir):
        """Parse Abaqus HTML documentation and store chunks in the vector DB.

        Walks *docs_dir* for .htm/.html files, splits by <h2>/<h3> headings,
        and inserts each chunk as a document.

        Args:
            docs_dir: Root directory containing Abaqus HTML documentation.

        Returns:
            Number of chunks ingested.
        """
        from bs4 import BeautifulSoup

        if not os.path.isdir(docs_dir):
            logger.error("Documentation directory not found: %s", docs_dir)
            return 0

        chunks = []
        ids = []
        metadatas = []
        count = 0

        for root, _dirs, files in os.walk(docs_dir):
            for fname in files:
                if not fname.lower().endswith((".htm", ".html")):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                except Exception as e:
                    logger.debug("Skipping %s: %s", fpath, e)
                    continue

                # Split by headings
                page_chunks = self._split_by_headings(soup, fpath)
                for title, text, meta in page_chunks:
                    if len(text.strip()) < 30:
                        continue
                    doc_id = f"html_{count}"
                    chunks.append(text[:2000])  # cap chunk size
                    ids.append(doc_id)
                    metadatas.append(meta)
                    count += 1

        if chunks:
            # Insert in batches of 500
            for i in range(0, len(chunks), 500):
                self._collection.add(
                    documents=chunks[i:i + 500],
                    ids=ids[i:i + 500],
                    metadatas=metadatas[i:i + 500],
                )
            logger.info("Ingested %d HTML chunks from %s", count, docs_dir)

        return count

    def _split_by_headings(self, soup, filepath):
        """Split an HTML page into chunks by <h2>/<h3> headings.

        Returns list of (title, text, metadata) tuples.
        """
        heading_tags = soup.find_all(["h2", "h3"])
        rel_path = os.path.basename(filepath)

        if not heading_tags:
            # No headings — treat entire page as one chunk
            text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string if soup.title else rel_path
            return [(title, text, {"source": rel_path, "title": str(title)})]

        chunks = []
        for i, heading in enumerate(heading_tags):
            title = heading.get_text(strip=True)
            # Collect text until next heading
            parts = [title]
            sibling = heading.find_next_sibling()
            while sibling and sibling.name not in ("h2", "h3"):
                text = sibling.get_text(separator="\n", strip=True)
                if text:
                    parts.append(text)
                sibling = sibling.find_next_sibling()

            full_text = "\n".join(parts)

            # Try to extract class/method name from title
            class_name = ""
            method_name = ""
            m = re.search(r"(\w+)\.(\w+)", title)
            if m:
                class_name = m.group(1)
                method_name = m.group(2)

            meta = {
                "source": rel_path,
                "title": title,
                "class_name": class_name,
                "method_name": method_name,
            }
            chunks.append((title, full_text, meta))

        return chunks

    # ------------------------------------------------------------------
    # Ingestion: manual JSON docs
    # ------------------------------------------------------------------

    def ingest_manual_docs(self, docs_list):
        """Add manually curated API documentation entries.

        Args:
            docs_list: List of dicts with keys: title, category, content,
                and optionally example.

        Returns:
            Number of entries ingested.
        """
        if not docs_list:
            return 0

        chunks = []
        ids = []
        metadatas = []

        for i, entry in enumerate(docs_list):
            title = entry.get("title", f"Manual entry {i}")
            content = entry.get("content", "")
            example = entry.get("example", "")
            category = entry.get("category", "general")

            text = f"## {title}\n{content}"
            if example:
                text += f"\n\n### Example\n{example}"

            doc_id = f"manual_{i}"
            chunks.append(text[:2000])
            ids.append(doc_id)
            metadatas.append({
                "source": "manual",
                "title": title,
                "category": category,
            })

        self._collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info("Ingested %d manual doc entries", len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    # Chinese → English keyword mapping for better retrieval with multilingual model
    _ZH_EN_MAP = {
        "接触": "contact friction surface-to-surface",
        "摩擦": "friction contact tangential penalty",
        "绑定": "tie constraint bond",
        "热传导": "heat transfer thermal conductivity steady-state",
        "热分析": "heat transfer thermal analysis",
        "温度": "temperature heat transfer thermal conductivity",
        "稳态": "steady-state heat transfer",
        "瞬态": "transient heat transfer",
        "导热": "conductivity heat transfer thermal",
        "对流": "convection film condition heat transfer",
        "屈曲": "buckling stability critical load eigenvalue",
        "失稳": "buckling stability critical",
        "稳定性": "buckling stability",
        "临界载荷": "critical load buckling eigenvalue",
        "螺栓": "bolt pretension load",
        "预紧": "bolt pretension",
        "弹簧": "spring dashpot stiffness element",
        "阻尼": "dashpot damping element",
        "刚体": "rigid body constraint reference point",
        "耦合": "coupling constraint distributing",
        "幅值": "amplitude time-varying tabular load",
        "载荷随时间": "amplitude time-varying tabular load",
        "热膨胀": "thermal expansion coupled temperature displacement",
        "热力耦合": "coupled temperature displacement thermal mechanical",
        "热应力": "thermal stress expansion temperature predefined",
        "内聚力": "cohesive zone delamination traction separation",
        "分层": "delamination cohesive",
        "脱粘": "delamination debond cohesive",
        "子模型": "submodel submodeling global local",
        "连接器": "connector beam hinge wire",
        "多点约束": "MPC multi-point constraint beam pin",
        "碰撞": "contact impact collision explicit general",
        "通用接触": "general contact explicit all-with-self",
        "参考点": "reference point coupling rigid body",
    }

    def _get_english_keywords(self, query):
        """Extract English keyword string from Chinese query via mapping."""
        extra = []
        for zh, en in self._ZH_EN_MAP.items():
            if zh in query:
                extra.append(en)
        return " ".join(extra)

    def _augment_query(self, query):
        """Augment Chinese query with English keywords for better embedding match."""
        en = self._get_english_keywords(query)
        if en:
            return query + " " + en
        return query

    def retrieve(self, query, top_k=5):
        """Semantic search — return the most relevant document chunks.

        Uses dual-query strategy: searches with both the augmented original
        query and pure English keywords, then merges results by best distance.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: title, content, source, distance.
        """
        if self._collection.count() == 0:
            return []

        n = min(top_k * 2, self._collection.count())

        # Query 1: augmented original (Chinese + English)
        augmented = self._augment_query(query)
        results1 = self._collection.query(query_texts=[augmented], n_results=n)

        # Query 2: pure English keywords (avoids Chinese text diluting the match)
        en_keywords = self._get_english_keywords(query)

        # Merge results — deduplicate by title, keep lowest distance
        seen = {}  # title -> doc dict
        for i, doc_text in enumerate(results1["documents"][0]):
            meta = results1["metadatas"][0][i]
            dist = results1["distances"][0][i] if results1.get("distances") else None
            title = meta.get("title", "Untitled")
            if title not in seen or (dist is not None and dist < seen[title]["distance"]):
                seen[title] = {
                    "title": title,
                    "content": doc_text,
                    "source": meta.get("source", "unknown"),
                    "distance": dist,
                }

        if en_keywords:
            results2 = self._collection.query(query_texts=[en_keywords], n_results=n)
            for i, doc_text in enumerate(results2["documents"][0]):
                meta = results2["metadatas"][0][i]
                dist = results2["distances"][0][i] if results2.get("distances") else None
                title = meta.get("title", "Untitled")
                if title not in seen or (dist is not None and dist < seen[title]["distance"]):
                    seen[title] = {
                        "title": title,
                        "content": doc_text,
                        "source": meta.get("source", "unknown"),
                        "distance": dist,
                    }

        # Sort by distance, filter, and return top_k
        docs = sorted(seen.values(), key=lambda d: d["distance"] or 999)
        docs = [d for d in docs if d["distance"] is None or d["distance"] <= _MAX_DISTANCE]

        # Prefer manual docs over HTML noise: if manual docs exist in results,
        # demote HTML docs that aren't clearly Abaqus-scripting related
        has_manual = any(d["source"] == "manual" for d in docs)
        if has_manual:
            docs = [d for d in docs
                    if d["source"] == "manual"
                    or d["distance"] is not None and d["distance"] < 0.55]

        return docs[:top_k]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_indexed(self):
        """Check whether the vector DB has any documents."""
        return self._collection.count() > 0

    def count(self):
        """Return number of documents in the collection."""
        return self._collection.count()
