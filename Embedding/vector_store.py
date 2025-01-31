import os
import time
import logging
from uuid import uuid4

from typing import List, Tuple
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, HnswConfigDiff, Modifier, OptimizersConfigDiff, Fusion, FusionQuery, Prefetch

# Example placeholders (replace with your own enums/classes)
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from fastembed import SparseTextEmbedding

# Example placeholders for config references
class QdrantConfigInfo:
    """
    Placeholder for Qdrant configuration settings.
    """
    # Store in current directory
    QDRANT_SERVICE_FULL_ADDRESS="http://qdrantdb:6333"
    QDRANT_DEFAULT_DENSE_VECTOR_NAME = "dense"
    QDRANT_DEFAULT_SPARSE_VECTOR_NAME = "sparse"
    QDRANT_DEFAULT_PAYLOAD_CONTENT_NAME = "content"
    QDRANT_DEFAULT_PAYLOAD_METADATA_NAME = "metadata"
    QDRANT_ON_DISK_PAYLOAD = True
    QDRANT_UPLOAD_BATCH_SIZE = 64
    QDRANT_UPLOAD_PARALLEL = 1
    QDRANT_UPLOAD_MAX_RETRIES = 3
    QDRANT_GLOBAL_DENSE_INDEXING_CONFIG = {
        "distance": "Cosine",
        "on_disk": True,
        "m": 32,
        "ef_construct": 64,
        "max_indexing_threads": 4,
        "payload_m": 0,
    }
    QDRANT_GLOBAL_SPARSE_INDEXING_CONFIG = {
        "on_disk": True,
    }
    QDRANT_OPTIMISER_SETTINGS = {
        "indexing_threshold_kb": 20000,
        "indexing_threshold_kb_during_uploading_docs": 0,
    }
    QDRANT_DELAY_TO_CHECK_COLLECTION_INDEXED = 5

# Just an example for retrieval limit
class RetrievalInfo:
    """
    Placeholder for retrieval settings.
    """
    QDRANT_DENSE_PREFETCH_LIMIT = 10
    QDRANT_SPARSE_PREFETCH_LIMIT = 10

# datatype class for metadata
class Metadata:
    """
    Placeholder for metadata class.
    
    Fields:
        source: str
        row_no: int
        sheet: str
        document_id: str
    """
    def __init__(self, source: str, sheet: str, row_no: int, document_id: str):
        self.source = source
        self.row_no = row_no
        self.sheet = sheet
        self.document_id = document_id
        
    def to_dict(self):
        """
        Convert metadata to dictionary.
        """
        return {
            "source": self.source,
            "row_no": self.row_no,
            "sheet": self.sheet,
            "document_id": self.document_id,
        }
        
    @classmethod
    def from_dict(cls, metadata_dict):
        """
        Create metadata from dictionary.
        """
        return cls(
            source=metadata_dict.get("source", ""),
            sheet=metadata_dict.get("sheet", ""),
            row_no=metadata_dict.get("row_no", 0),
            document_id=metadata_dict.get("document_id", ""),
        )
        
class QdrantStore:
    """
    Minimal Qdrant DB Repository exposing only:
      1) __init__ (constructor)
      2) ingest_data (ingestion pipeline)
      3) retrieve_data (query pipeline)
    """

    def __init__(self):
        """Initialize client and embedding models."""
        self.logger = logging.getLogger("QdrantLogger")
        
        # 1) Connect to Qdrant
        self.client = None

        # 2) Initialize embedding models (example usage)
        self.dense_embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        self.sparse_embedding_model = SparseTextEmbedding(
            model_name="Qdrant/bm25",
            threads=4,
        )

    def ingest_data(self, collection_name: str, documents: List[Document]) -> bool:
        """
        Ingest (add) documents into Qdrant.
        
        Steps:
          - Create collection (if doesn't exist)
          - Disable indexing (for faster bulk upload)
          - Compute dense/sparse embeddings if not already provided
          - Convert to Qdrant points and upload
          - Re-enable indexing and confirm indexing is finished
          
        Args:
            collection_name (str): Name of the collection to ingest into.
            documents (List[Document]): List of documents to ingest.
            
        Returns:
            bool: True if ingestion is successful, False otherwise.
        """
        try:
            # 1) Create collection if needed
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        QdrantConfigInfo.QDRANT_DEFAULT_DENSE_VECTOR_NAME: VectorParams(
                            size=len(self.dense_embedding_model.embed_query("test")),
                            distance=Distance.COSINE,
                            on_disk=QdrantConfigInfo.QDRANT_GLOBAL_DENSE_INDEXING_CONFIG["on_disk"],
                            hnsw_config=HnswConfigDiff(
                                m=QdrantConfigInfo.QDRANT_GLOBAL_DENSE_INDEXING_CONFIG["m"],
                                payload_m=QdrantConfigInfo.QDRANT_GLOBAL_DENSE_INDEXING_CONFIG["payload_m"],
                                ef_construct=QdrantConfigInfo.QDRANT_GLOBAL_DENSE_INDEXING_CONFIG["ef_construct"],
                                on_disk=QdrantConfigInfo.QDRANT_GLOBAL_DENSE_INDEXING_CONFIG["on_disk"],
                                max_indexing_threads=QdrantConfigInfo.QDRANT_GLOBAL_DENSE_INDEXING_CONFIG["max_indexing_threads"],
                            ),
                        )
                    },
                    sparse_vectors_config={
                        QdrantConfigInfo.QDRANT_DEFAULT_SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(
                                on_disk=QdrantConfigInfo.QDRANT_GLOBAL_SPARSE_INDEXING_CONFIG["on_disk"],
                            ),
                            modifier=Modifier.IDF,
                        )
                    },
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=QdrantConfigInfo.QDRANT_OPTIMISER_SETTINGS["indexing_threshold_kb"]
                    ),
                    on_disk_payload=QdrantConfigInfo.QDRANT_ON_DISK_PAYLOAD,
                )

            # 2) Temporarily disable indexing for faster bulk uploads
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=QdrantConfigInfo.QDRANT_OPTIMISER_SETTINGS["indexing_threshold_kb_during_uploading_docs"]
                ),
            )

            # 3) Prepare data to upload
            processed_documents = []
            sparse_embeddings = []
            payloads = []
            ids = [str(uuid4()) for _ in range(len(documents))]

            for idx, doc in enumerate(documents):
                doc_text = doc.page_content
                metadata = doc.metadata
                metadata["document_id"] = ids[idx]
                processed_documents.append(doc_text)
                payloads.append({
                    QdrantConfigInfo.QDRANT_DEFAULT_PAYLOAD_METADATA_NAME: metadata,
                    QdrantConfigInfo.QDRANT_DEFAULT_PAYLOAD_CONTENT_NAME: doc_text,
                })
                    
            # Compute dense for all docs in one go
            dense_embeddings = self.dense_embedding_model.embed_documents(texts=processed_documents)
                
            # Compute sparse for all docs in one go
            sparse_embeds = self.sparse_embedding_model.embed(documents=processed_documents)
            for sparse_embed in sparse_embeds:
                # Convert fastembed sparse => Qdrant sparse
                sparse_embeddings.append(sparse_embed.as_object())

            # 4) Convert to Qdrant Points
            points = []
            for i in range(len(processed_documents)):
                points.append(
                    models.PointStruct(
                        id=ids[i],
                        vector={
                            QdrantConfigInfo.QDRANT_DEFAULT_DENSE_VECTOR_NAME: dense_embeddings[i],
                            QdrantConfigInfo.QDRANT_DEFAULT_SPARSE_VECTOR_NAME: sparse_embeddings[i],
                        },
                        payload=payloads[i],
                    )
                )

            # 5) Upload points
            self.client.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=QdrantConfigInfo.QDRANT_UPLOAD_BATCH_SIZE,
                parallel=QdrantConfigInfo.QDRANT_UPLOAD_PARALLEL,
                max_retries=QdrantConfigInfo.QDRANT_UPLOAD_MAX_RETRIES,
            )

            # 6) Re-enable indexing
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=QdrantConfigInfo.QDRANT_OPTIMISER_SETTINGS["indexing_threshold_kb"]
                ),
            )

            # 7) Wait until status is green (indexing finished)
            while True:
                status = self.client.get_collection(collection_name).status.value
                if status == "green":
                    break
                elif status == "red":
                    self.logger.error("Collection %s is in red status, stopping...", collection_name)
                    break
                time.sleep(QdrantConfigInfo.QDRANT_DELAY_TO_CHECK_COLLECTION_INDEXED)

            self.logger.info("Ingested %d documents into collection: %s", len(documents), collection_name)
            return True
        except Exception as e:
            self.logger.error("Error during ingestion: %s", str(e))
            return False

    def retrieve_data(self, collection_name: str, queries: List[str], n_results: int, top_k: int) -> List[Document]:
        """
        Retrieve data from Qdrant with hybrid (dense + sparse) RRF fusion.
        For each query:
          - Generate dense + sparse embeddings
          - Perform multi-vector search with RRF fusion
          - Convert results back to DocumentDto
          
        Args:
            collection_name (str): Name of the collection to query.
            queries (List[str]): List of queries to search.
            n_results (int): Number of results to retrieve per query.
            top_k (int): Number of results to return after re-ranking (Final top-k, useful for multi-query retrieval/hyde based decomposition).
        
        Returns:
            List[Document]: List of retrieved documents.
        """

        # Quick check
        if not self.client.collection_exists(collection_name):
            self.logger.warning("Collection '%s' not found.", collection_name)
            return []

        # Gather final results
        all_results: List[Document] = []

        for query_text in queries:
            # 1) Compute dense & sparse embeddings
            dense_vec = self.dense_embedding_model.embed_query(query_text)
            sparse_vectors = self.sparse_embedding_model.embed(query_text)
            for sparse_vec in sparse_vectors:
                sparse_vec = sparse_vec.as_object()

            # 2) Create RRF-Fusion prefetch
            prefetch = [
                Prefetch(
                    query=dense_vec,
                    using=QdrantConfigInfo.QDRANT_DEFAULT_DENSE_VECTOR_NAME,
                    limit=RetrievalInfo.QDRANT_DENSE_PREFETCH_LIMIT,
                ),
                Prefetch(
                    query=models.SparseVector(**sparse_vec),
                    using=QdrantConfigInfo.QDRANT_DEFAULT_SPARSE_VECTOR_NAME,
                    limit=RetrievalInfo.QDRANT_SPARSE_PREFETCH_LIMIT,
                ),
            ]

            # 3) Run query
            response = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),  # Use RRF as example
                with_payload=True,
                limit=n_results,
            )

            # 4) Convert each result to DocumentDto
            for point in response.points:
                payload = point.payload
                metadata_dict = Metadata.from_dict(payload.get(QdrantConfigInfo.QDRANT_DEFAULT_PAYLOAD_METADATA_NAME, {})).to_dict()
                doc_text = payload.get(QdrantConfigInfo.QDRANT_DEFAULT_PAYLOAD_CONTENT_NAME, "")
                
                # Build DocumentDto (assuming your MetadataDto)
                doc = Document(page_content=doc_text, metadata=metadata_dict)
                all_results.append(doc)

        # Rerank using RRF
        all_results = self._rerank_retrieved_documents_using_rrf(
            retrieved_documents=all_results,
            n_queries=len(queries),
            n_docs_per_query=n_results,
            top_k=top_k,
        )
        return all_results
    
    def get_documents_count_in_collection(self, collection_name: str) -> int:
        """
        Get the total number of documents in a collection.
        
        Args:
            collection_name (str): Name of the collection.
            
        Returns:
            int: Number of documents in the collection.
        """
        if not self.client.collection_exists(collection_name):
            self.logger.warning("Collection '%s' not found.", collection_name)
            return 0

        return self.client.count(collection_name=collection_name).count

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name (str): Name of the collection to delete.
            
        Returns:
            bool: True if collection is deleted, False otherwise.
        """
        is_deleted = self.client.delete_collection(collection_name=collection_name)
        self.logger.info("Deleted collection: %s", collection_name)
        return is_deleted
    
    def get_available_collections_info(self) -> Tuple[List[str], List[int]]:
        """
        Get the list of available collections.
        
        Returns:
            Tuple[List[str], List[int]]: Tuple of collection names and their respective document counts.
        """
        # Get all collections
        collections = self.client.get_collections().collections
        collections = [collection.name for collection in collections]
        
        # get counts of no of documents in each collection
        count_of_docs = []
        for collection in collections:
            count_of_docs.append(self.get_documents_count_in_collection(collection))
            
        # return list of collection names with count of documents
        return (collections, count_of_docs)
    
    def _rerank_retrieved_documents_using_rrf(
        self,
        retrieved_documents: List[Document],
        n_queries: int,
        n_docs_per_query: int,
        top_k: int,
    ) -> List[Document]:
        """
        Helper method to Re-rank the retrieved documents using RRF (Reciprocal Rank Fusion),
        then return:
        (1) n_queries, 
        (2) the re-ranked documents (up to TOP_K_DOCUMENTS_TO_RETURN),
        (3) the final number of returned documents.

        Args:
            retrieved_documents (List[Document]): Retrieved documents from the initial retrieval.
            n_queries (int): Number of queries.
            n_docs_per_query (int): Number of documents per query.
            top_k (int): Number of documents to return.
            
        Returns:
            List[Document]: Re-ranked documents.
        """
        
        try:
            self.logger.info("Re-ranking retrieved documents using RRF | Retrieved documents: %d | Top-K to return: %s", len(retrieved_documents), top_k)
            
            # Validation
            assert len(retrieved_documents) % n_queries <= n_docs_per_query, f"The number of retrieved documents {len(retrieved_documents)} does not match n_queries {n_queries} * n_docs_per_query {n_docs_per_query}." 
            # What if collection has less documents than required for RRF
       
            # Default case: only 1 query => no re-ranking needed
            if n_queries == 1:
                final_docs = retrieved_documents[:top_k]
                return final_docs

            # Compute RRF
            rankings = {}
            k = 60  # RRF constant
            for index, doc in enumerate(retrieved_documents):
                # Documents from the same query appear in consecutive blocks
                rank = (index % n_docs_per_query) + 1  # +1 so rank starts at 1
                score = 1 / (rank + k)
                rankings[doc.metadata["document_id"]] = rankings.get(doc.metadata["document_id"], 0) + score

            # Print rankings
            self.logger.debug("Rankings: %s", rankings)
            
            # Sort by cumulative RRF score (descending)
            sorted_doc_ids = sorted(rankings, key=rankings.get, reverse=True)
            
            # Build a mapping of doc_id -> actual DocumentDto
            doc_id_to_doc = {doc.metadata["document_id"]: doc for doc in retrieved_documents}

            # Create re-ranked list
            reranked_documents = [doc_id_to_doc[doc_id] for doc_id in sorted_doc_ids]

            # Slice to top_k
            final_docs = reranked_documents[:top_k]
            self.logger.info("Re-ranking completed | Final documents: %d", len(final_docs))
            return final_docs        
        except Exception as e:
            self.logger.error("Error during RRF re-ranking: %s", str(e))
            return []
    
    def close_connection(self) -> bool:
        """
        Close the connection to Qdrant.
        
        Returns:
            bool: True if connection is closed, False otherwise.
        """
        try:
            if self.client:
                self.client.close()
            self.logger.info("Connection to Qdrant closed.")
            return True
        except Exception as e:
            self.logger.error("Error during closing connection: %s", str(e))
            return False
    
    def reconnect_or_open_connection(self) -> bool:
        """
        Open the connection to Qdrant.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            self.client = QdrantClient(
                url=QdrantConfigInfo.QDRANT_SERVICE_FULL_ADDRESS
            )
            collections = self.client.get_collections()
            self.logger.info("Connection to Qdrant opened.")
            return True
        except Exception as e:
            self.logger.error("Error during reconnection: %s", str(e))
            return False
