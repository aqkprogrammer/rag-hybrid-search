"""Composition root: builds every component from `Settings` exactly once."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.config import Settings
from hybrid_rag.embeddings import Embedder, build_embedder
from hybrid_rag.ingestion import IngestionService, StructureAwareChunker
from hybrid_rag.llm import LLMProvider, build_llm
from hybrid_rag.logging_setup import get_logger
from hybrid_rag.rag import RAGService
from hybrid_rag.retrieval import HybridRetriever, Reranker, build_reranker
from hybrid_rag.storage import BM25Index, DocStore
from hybrid_rag.text import get_token_counter
from hybrid_rag.vectorstores import VectorStore, build_vector_store
from hybrid_rag.verification import CitationVerifier

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    docstore: DocStore
    bm25: BM25Index
    embedder: Embedder
    vector_store: VectorStore
    reranker: Reranker
    retriever: HybridRetriever
    llm: LLMProvider
    verifier: CitationVerifier
    ingestion: IngestionService
    rag: RAGService

    @classmethod
    def build(cls, settings: Settings, llm: LLMProvider | None = None) -> Container:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        docstore = DocStore(settings.sqlite_path)
        bm25 = BM25Index(docstore)
        embedder = build_embedder(settings)
        vector_store = build_vector_store(settings, embedder.dim, embedder.slug)
        reranker = build_reranker(settings)
        llm = llm or build_llm(settings)
        chunker = StructureAwareChunker(
            get_token_counter(settings.tokenizer, settings.tiktoken_encoding),
            chunk_size=settings.chunk_size_tokens,
            overlap=settings.chunk_overlap_tokens,
            min_tokens=settings.chunk_min_tokens,
        )
        retriever = HybridRetriever(settings, docstore, bm25, vector_store, embedder, reranker)
        verifier = CitationVerifier(
            embedder,
            threshold=settings.verifier_threshold,
            lexical_weight=settings.verifier_lexical_weight,
            judge=llm if settings.verifier_llm_judge else None,
        )
        ingestion = IngestionService(
            docstore, vector_store, embedder, chunker, batch_size=settings.embedding_batch_size
        )
        rag = RAGService(settings, retriever, llm, verifier)
        return cls(
            settings=settings,
            docstore=docstore,
            bm25=bm25,
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            retriever=retriever,
            llm=llm,
            verifier=verifier,
            ingestion=ingestion,
            rag=rag,
        )

    def startup(self) -> None:
        """Rebuild derived indexes and optionally seed a demo corpus."""
        chunks = self.docstore.count_chunks()
        vectors = self.vector_store.count()
        if chunks and vectors != chunks:
            # e.g. the embedding model changed (new collection) or vectors were wiped.
            log.warning("vector_index_out_of_sync", chunks=chunks, vectors=vectors)
            self.ingestion.reindex()
        self.bm25.rebuild()
        seed_dir = self.settings.auto_seed_dir
        if seed_dir and self.docstore.count_documents() == 0:
            if seed_dir.is_dir():
                results = self.ingestion.ingest_directory(seed_dir)
                log.info("auto_seed_done", directory=str(seed_dir), documents=len(results))
            else:
                log.warning("auto_seed_dir_missing", directory=str(seed_dir))
        log.info(
            "startup_complete",
            documents=self.docstore.count_documents(),
            chunks=self.docstore.count_chunks(),
            vector_store=self.vector_store.backend,
            embedder=self.embedder.model_name,
            reranker=self.reranker.name,
            llm=f"{self.llm.name}:{self.llm.model}",
        )

    async def aclose(self) -> None:
        await self.llm.aclose()
        self.vector_store.close()
        self.docstore.close()
