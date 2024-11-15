# RAG-Application
End-to-end pipline for Retreival Augmented Generation leveraging hugging face open source LLMs and FAISS Vector store DB
The pipeline has following parts :- 
1. Loading the document
2. Document cleaning and preprocessing
3. Document Chunking
4. Storing chunk embeddings in vector store
5. Indexing
6. Retrieval


![RAG Pipeline Diagram](https://github.com/kgourav148/RAG-Application/blob/main/RAG%20Architecture.png)


## Scope of Improvement
1. Vector storage can augmented with Graph DB
2. Hierarchical Indexing
3. Dynamic Chunking can be adopted and chunking overlap tunning
4. Introduction of Re-ranking in retrieval process
5. Query transformation - Decomposition, Stepback, etc. 
