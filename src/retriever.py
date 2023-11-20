from typing import List, Dict, Tuple
import time
import tqdm
import uuid
import numpy as np
import torch
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search.lexical.elastic_search import ElasticSearch
from .bing import search_bing_batch


def get_random_doc_id():
    return f'_{uuid.uuid4()}'


class SearchEngineConnector:
    def __init__(
        self,
        engine: str,
        only_domain: str = None,
        exclude_domains: List[str] = [],
    ):
        assert engine in {'bing'}
        self.engine = engine
        self.only_domain = only_domain
        self.exclude_domains = exclude_domains

    def retrieve(
        self,
        corpus = None,
        queries: Dict[int, str] = None,
        **kwargs,
    ):
        qs = list(queries.values())
        if self.engine == 'bing':
            all_results = search_bing_batch(
                qs, only_domain=self.only_domain, exclude_domains=self.exclude_domains)
        else:
            raise NotImplementedError
        qid2results: Dict[int, Dict[str, Tuple[float, str]]] = {}
        for (qid, query), results in zip(queries.items(), all_results):
            qid2results[qid] = {str(r['url']) + get_random_doc_id(): (0, r['snippet']) for r in results}
        return qid2results


class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        index_name: str = None,
        engine: str = 'elasticsearch',
        **search_engine_kwargs,
    ):
        self.tokenizer = tokenizer
        # load index
        assert engine in {'elasticsearch', 'bing'}
        if engine == 'elasticsearch':
            self.max_ret_topk = 1000
            self.retriever = EvaluateRetrieval(
                BM25Search(index_name=index_name, hostname='localhost', initialize=False, number_of_shards=1),
                k_values=[self.max_ret_topk])
        else:
            self.max_ret_topk = 50
            self.retriever = SearchEngineConnector(engine, **search_engine_kwargs)

    def retrieve(
        self,
        queries: List[str],  # (bs,)
        filter_ids: List[str] = None,  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
    ):
        assert topk <= self.max_ret_topk
        device = None
        bs = len(queries)

        # truncate queries
        if max_query_length:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            # truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # retrieve
        filter_ids = filter_ids or ([None] * len(queries))
        results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), list(zip(queries, filter_ids)))), disable_tqdm=True)

        # prepare outputs
        docids: List[str] = []
        docs: List[str] = []
        for qid, query in enumerate(queries):
            _docids: List[str] = []
            _docs: List[str] = []
            if qid in results:
                for did, (score, text) in results[qid].items():
                    _docids.append(did)
                    _docs.append(text)
                    if len(_docids) >= topk:
                        break
            if len(_docids) < topk:  # add dummy docs
                _docids += [get_random_doc_id() for _ in range(topk - len(_docids))]
                _docs += [''] * (topk - len(_docs))
            docids.extend(_docids)
            docs.extend(_docs)

        docids = np.array(docids).reshape(bs, topk)  # (bs, topk)
        docs = np.array(docs).reshape(bs, topk)  # (bs, topk)
        return docids, docs


def bm25search_search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, Tuple[str, str]], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
    # Index the corpus within elastic-search
    # False, if the corpus has been already indexed
    if self.initialize:
        self.index(corpus)
        # Sleep for few seconds so that elastic-search indexes the docs properly
        time.sleep(self.sleep_for)

    #retrieve results from BM25
    query_ids = list(queries.keys())
    filter_ids = [queries[qid][1] for qid in query_ids]
    queries = [queries[qid][0] for qid in query_ids]

    final_results: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que', disable=kwargs.get('disable_tqdm', False)):
        query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
        results = self.es.lexical_multisearch(
            texts=queries[start_idx:start_idx+self.batch_size],
            filter_ids=filter_ids[start_idx:start_idx+self.batch_size],
            top_hits=top_k)
        for (query_id, hit) in zip(query_ids_batch, results):
            scores = {}
            for corpus_id, score, text in hit['hits']:
                scores[corpus_id] = (score, text)
                final_results[query_id] = scores

    return final_results

BM25Search.search = bm25search_search


def elasticsearch_lexical_multisearch(self, texts: List[str], filter_ids: List[str] = None, top_hits: int = 10, skip: int = 0) -> Dict[str, object]:
    """Multiple Query search in Elasticsearch

    Args:
        texts (List[str]): Multiple query texts
        top_hits (int): top k hits to be retrieved
        skip (int, optional): top hits to be skipped. Defaults to 0.

    Returns:
        Dict[str, object]: Hit results
    """
    request = []

    assert skip + top_hits <= 10000, "Elastic-Search Window too large, Max-Size = 10000"

    filter_ids = filter_ids or ([None] * len(texts))
    for text, fid in zip(texts, filter_ids):
        req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
        if fid is not None:
            req_body = {
                "_source": True, # No need to return source objects
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": text,  # matching query with both text and title fields
                                "type": "best_fields",
                                "fields": [self.title_key, self.text_key],
                                "tie_breaker": 0.5
                            },
                        },
                        "filter": {
                            "term": {
                                "_id": fid
                            }
                        }
                    },
                },
                "size": skip + top_hits, # The same paragraph will occur in results
            }
        else:
            req_body = {
                "_source": True, # No need to return source objects
                "query": {
                    "multi_match": {
                        "query": text, # matching query with both text and title fields
                        "type": "best_fields",
                        "fields": [self.title_key, self.text_key],
                        "tie_breaker": 0.5
                    }
                },
                "size": skip + top_hits, # The same paragraph will occur in results
            }
        request.extend([req_head, req_body])

    res = self.es.msearch(body = request)

    result = []
    for resp in res["responses"]:
        responses = resp["hits"]["hits"][skip:] if 'hits' in resp else []

        hits = []
        for hit in responses:
            hits.append((hit["_id"], hit['_score'], hit['_source']['txt']))

        result.append(self.hit_template(es_res=resp, hits=hits))
    return result

ElasticSearch.lexical_multisearch = elasticsearch_lexical_multisearch


def elasticsearch_hit_template(self, es_res: Dict[str, object], hits: List[Tuple[str, float]]) -> Dict[str, object]:
    """Hit output results template

    Args:
        es_res (Dict[str, object]): Elasticsearch response
        hits (List[Tuple[str, float]]): Hits from Elasticsearch

    Returns:
        Dict[str, object]: Hit results
    """
    result = {
        'meta': {
            'total': es_res['hits']['total']['value'] if 'hits' in es_res else None,
            'took': es_res['took'] if 'took' in es_res else None,
            'num_hits': len(hits)
        },
        'hits': hits,
    }
    return result

ElasticSearch.hit_template = elasticsearch_hit_template
