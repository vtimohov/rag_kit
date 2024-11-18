import chromadb
import cohere

class Retrivel ():
  def __init__ (self, ch_collection_name:str):
    self.co = cohere.ClientV2()
    ch_client = chromadb.Client()
    self.ch_collection = ch_client.get_collection(name=ch_collection_name)
    


  def _vector_search (self, query):
    result = self.ch_collection.query(
        query_texts=query,
        n_results=20,
        include = ['documents'],
        )

    return result['documents'][0]

  def _reranking (self, query, documents, n_chunks) -> list[int]:
    _top_n = n_chunks if n_chunks else len(documents)

    response = self.co.rerank(
      model="rerank-english-v3.0",
      query=query,
      documents=documents,
      top_n=_top_n,
      )

    return [i.dict()['index'] for i in response.results]

  def _build_context (self, documents, reranks) -> str:
    top_ranked_documents =  [documents[i] for i in reranks]
    return "".join(top_ranked_documents)

  def retrive (self, query:str, n_chunks: int = None):
    retrived_documents = self._vector_search(query)
    reranks = self._reranking(query, retrived_documents, n_chunks)
    context = self._build_context(retrived_documents, reranks)
    
    return context
    