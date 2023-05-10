from typing import List, Dict, Tuple
import time
import tqdm
import uuid
import numpy as np
import torch
from filelock import FileLock
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
        file_lock: FileLock = None,
    ):
        self.fake_score = 0
        assert engine in {'bing'}
        self.engine = engine
        self.only_domain = only_domain
        self.exclude_domains = exclude_domains
        self.file_lock = file_lock

    def retrieve(
        self,
        corpus = None,
        queries: Dict[int, str] = None,
        **kwargs,
    ):
        if self.file_lock is not None:
            self.file_lock.acquire()
        try:
            qs = list(queries.values())
            if self.engine == 'bing':
                se_results = search_bing_batch(
                    qs, only_domain=self.only_domain, exclude_domains=self.exclude_domains)
            else:
                raise NotImplementedError
            results: Dict[int, Dict[str, Tuple[float, str]]] = {}
            for (qid, query), ser in zip(queries.items(), se_results):
                results[qid] = {str(r['url']) + get_random_doc_id(): (self.fake_score, r['snippet']) for did, r in enumerate(ser)}
            return results
        finally:
            if self.file_lock is not None:
                self.file_lock.release()


class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        collator = None,
        dataset: GenericDataLoader = None,
        index_name: str = None,
        engine: str = 'elasticsearch',
        encode_retrieval_in: str = 'encoder',
        use_encoder_input_ids: bool = False,
        use_decoder_input_ids: bool = True,
        file_lock: FileLock = None,
        **search_engine_kwargs,
    ):
        self.tokenizer = tokenizer
        self.collator = collator
        self.corpus, self.queries, self.qrels = dataset
        # load index
        assert engine in {'elasticsearch', 'bing'}
        if engine == 'elasticsearch':
            self.max_ret_topk = 1000
            self.retriever = EvaluateRetrieval(
                BM25Search(index_name=index_name, hostname='localhost', initialize=False, number_of_shards=1),
                k_values=[self.max_ret_topk])
        else:
            self.max_ret_topk = 50
            self.retriever = SearchEngineConnector(engine, file_lock=file_lock, **search_engine_kwargs)

        self.encode_retrieval_in = encode_retrieval_in
        assert encode_retrieval_in in {'encoder', 'decoder'}
        self.use_encoder_input_ids = use_encoder_input_ids
        self.use_decoder_input_ids = use_decoder_input_ids
        assert use_encoder_input_ids or use_decoder_input_ids, 'nothing used as queries'

    def retrieve_and_prepare(
        self,
        encoder_input_ids: torch.LongTensor = None,  # (bs, encoder_seq_len)
        decoder_input_ids: torch.LongTensor = None,  # (bs, decoder_seq_len)
        encoder_texts: List[str] = None,  # (bs, encoder_seq_len)
        decoder_texts: List[str] = None,  # (bs, encoder_seq_len)
        ctx_ids: np.ndarray = None,  # (bs, topk)
        qids: np.ndarray = None,  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
        use_gold: bool = False,
        joint_encode_retrieval: bool = False,
        merge_ctx: bool = False,
    ):
        assert topk <= self.max_ret_topk
        device = None
        if self.use_encoder_input_ids and encoder_input_ids is not None:
            device = encoder_input_ids.device
        if self.use_decoder_input_ids and decoder_input_ids is not None:
            device = decoder_input_ids.device

        if self.use_encoder_input_ids and encoder_input_ids is not None:
            encoder_texts: List[str] = self.tokenizer.batch_decode(encoder_input_ids, skip_special_tokens=True)
        if self.use_decoder_input_ids and decoder_input_ids is not None:
            decoder_texts: List[str] = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)

        if encoder_texts is not None:
            bs = len(encoder_texts)
        if decoder_texts is not None:
            bs = len(decoder_texts)

        if ctx_ids is not None:  # use doc ids passed in
            docids: List[str] = ctx_ids.reshape(-1)
            docs: List[str] = [self.corpus[did]['text'] for did in docids]
        elif use_gold:  # use qrels annotations to find gold ctxs
            docids: List[str] = []
            docs: List[str] = []
            for qid in qids:
                rel_dids = [did for did, r in self.qrels[qid].items() if r]
                rel_docs = [self.corpus[did]['text'] for did in rel_dids]
                if merge_ctx:
                    rel_dids = rel_dids[:1]
                    rel_docs = [' '.join(rel_docs)]
                assert len(rel_dids) == len(rel_docs) == topk, f'{len(rel_dids)} {len(rel_docs)} {topk}'
                docids.extend(rel_dids)
                docs.extend(rel_docs)
        else:
            # prepare queries
            queries: List[str] = []
            if self.use_encoder_input_ids and encoder_texts is not None:
                queries = list(encoder_texts)
            if self.use_decoder_input_ids and decoder_texts is not None:
                if queries:
                    assert len(queries) == len(decoder_texts), 'inconsistent length'
                    queries = [f'{q} {t}' for q, t in zip(queries, decoder_texts)]
                else:
                    queries = list(decoder_texts)

            # truncate queries
            if max_query_length:
                ori_ps = self.tokenizer.padding_side
                ori_ts = self.tokenizer.truncation_side
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
            #print('REAL QUERY:', queries[0])
            results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(self.corpus, dict(zip(range(len(queries)), queries)), disable_tqdm=True)

            # prepare outputs
            docids: List[str] = []
            docs: List[str] = []
            for qid, query in enumerate(queries):
                _docids = []
                _docs = []
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

        if device is None:
            docids = np.array(docids).reshape(bs, topk)  # (batch_size, topk)
            docs = np.array(docs).reshape(bs, topk)  # (batch_size, topk)
            return docids, docs

        # tokenize
        if joint_encode_retrieval:
            for i in range(bs):
                for j in range(topk):
                    if self.encode_retrieval_in == 'encoder':
                        docs[i * topk + j] = f'{docs[i * topk + j]}\n{encoder_texts[i]}'
                    elif self.encode_retrieval_in == 'decoder':
                        docs[i * topk + j] = f'{docs[i * topk + j]}\n'
                    else:
                        raise NotImplementedError
            if self.encode_retrieval_in == 'encoder':
                ctxs = self.collator.encode_context(docs, max_length=self.collator.max_context_len + self.collator.max_question_len)
            elif self.encode_retrieval_in == 'decoder':
                ctxs = self.collator.encode_context(docs)
                assert ctxs.input_ids[:, 0].eq(self.collator.get_real_decoder_start_token_id).all()
                assert topk == 1
                #decoder_input_ids = decoder_input_ids[:, 1:]  # skip decoder_start_token
                ctxs.input_ids = torch.cat([ctxs.input_ids.to(device), decoder_input_ids], 1)
                ctxs.attention_mask = torch.cat([ctxs.attention_mask.to(device), torch.ones_like(decoder_input_ids).to(ctxs.attention_mask.dtype)], 1)
            else:
                raise NotImplementedError
        else:
            ctxs = self.collator.encode_context(docs)
        input_ids = ctxs.input_ids.view(bs, topk, -1).to(device)  # (batch_size, topk, seq_length)
        attention_mask = ctxs.attention_mask.view(bs, topk, -1).to(device)  # (batch_size, topk, seq_length)
        docids = np.array(docids).reshape(bs, topk)  # (batch_size, topk)
        return docids, input_ids, attention_mask



def bm25search_search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
    # Index the corpus within elastic-search
    # False, if the corpus has been already indexed
    if self.initialize:
        self.index(corpus)
        # Sleep for few seconds so that elastic-search indexes the docs properly
        time.sleep(self.sleep_for)

    #retrieve results from BM25
    query_ids = list(queries.keys())
    queries = [queries[qid] for qid in query_ids]

    final_results: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que', disable=kwargs.get('disable_tqdm', False)):
        query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
        results = self.es.lexical_multisearch(
            texts=queries[start_idx:start_idx+self.batch_size],
            top_hits=top_k)
        for (query_id, hit) in zip(query_ids_batch, results):
            scores = {}
            for corpus_id, score, text in hit['hits']:
                scores[corpus_id] = (score, text)
                final_results[query_id] = scores

    return final_results

BM25Search.search = bm25search_search


def elasticsearch_lexical_multisearch(self, texts: List[str], top_hits: int, skip: int = 0) -> Dict[str, object]:
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

        for text in texts:
            req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
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

ElasticSearch.lexical_multisearch = elasticsearch_lexical_multisearch
ElasticSearch.hit_template = elasticsearch_hit_template
