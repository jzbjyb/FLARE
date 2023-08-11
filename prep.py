from typing import List, Tuple, Union, Dict
import argparse
import random
import json
import time
import glob
import csv
import evaluate
import re
import logging
from tqdm import tqdm
import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from src.datasets import WikiMultiHopQA, WikiAsp, ASQA
from src.utils import Utils


def eval(
    model: str,
    dataset: str,
    jsonl_files: List[str],
    anchor_text: List[str] = [],
    prefix_to_remove: List[str] = [],
    retrieval_percentiles: List[Union[int, float]] = [1, 0.25, 0.5, 0.75, 1.0],
    remove_followup: Tuple[str, str] = None,  # ('Follow up[^:]*:', '?'),
    beir_dir: str = None,
    consistency_suffix: str = 'run',
    use_multi_ref: bool = True,
    debug: bool = False,
):
    if not anchor_text:
        anchor_text = []
    anchor_text = anchor_text if type(anchor_text) is list else [anchor_text]
    if beir_dir is not None:
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='dev')
    else:
        corpus = queries = qrels = None

    def add_metric_kvs(metric_dict):
        for k, v in metric_dict.items():
            final_metrics[k] += v

    def choose_reference(example):
        if 'answers' in example and use_multi_ref:  # multiple
            return example['answers']
        return example['gold_output'] if 'gold_output' in example else example['output']

    def choose_final_answer(example):
        if 'answers' in example and use_multi_ref:  # multiple
            answers = example['answers']
        else:
            answers = [example['answer']]
        for i, answer in enumerate(answers):
            for pattern in prefix_to_remove + anchor_text[:1]:
                if not pattern:
                    continue
                find = re.compile(pattern).search(answer)
                if find:
                    answer = find.group(1)
                    answers[i] = answer
                    if dataset == 'strategyqa':
                        answer = answer.lower()
                        assert answer in {'yes', 'no'}
                    elif dataset == 'mmlu':
                        answer = answer.lower()
                        assert answer in {'a', 'b', 'c', 'd', 'e'}
        return answers if len(answers) > 1 else answers[0]

    def choose_full_prediction(example):
        if Utils.no_stop(model=model, dataset=dataset):
            pred = example['output'].strip()
        else:
            pred = example['output'].split('\n\n', 1)[0].strip()
        if prefix_to_remove:
            find = None
            for pattern in prefix_to_remove:
                find = re.compile(pattern).search(pred)
                if find:
                    pred = find.group(1)
                    break
            if find is None:
                logging.warning(f'format error "{pred}"')
        return pred

    def get_final_answer_from_pred(pred: str):
        final_ans = []
        for at in anchor_text:
            find = re.compile(at).search(pred)
            if find:
                final_ans.append(find.group(1))
                break
        return ' '.join(final_ans).strip()

    metric_func = evaluate.load('rouge')

    scount = 0
    search_per_example: List[int] = []
    final_metrics = {k: 0 for k in [
        'correct', 'incorrect', 'wrongformat',
        'f1', 'precision', 'recall',
        'ent_f1', 'ent_precision', 'ent_recall', 'num_ent',
        'avg_nll', 'ppl', 'tokens']}
    ret_accs: List[List[float]] = []
    ret_covers: List[List[float]] = []
    predictions: List[str] = []
    followups: List[str] = []
    references: List[str] = []
    num_steps: List[int] = []
    num_rets: List[int] = []
    retrieval_ratios: List[float] = []
    retrieval_hits: List[int] = []

    root_file = None
    if len(jsonl_files) > 1:  # consistency
        print(f'consistency of {len(jsonl_files)} files')
        for jf in jsonl_files:
            assert jf.rsplit('.', 1)[1].startswith(consistency_suffix)
        root_file = jsonl_files[0].rsplit('.', 1)[0]


    examples_all_files = [[json.loads(l) for l in open(jf)] for jf in jsonl_files]
    assert len(set([len(examples) for examples in examples_all_files])) == 1
    total = len(examples_all_files[0])

    consistency_examples: List[Dict] = []
    for i in tqdm(range(total)):
        examples: List[Dict] = [file[i] for file in examples_all_files]

        # aggregate multiple examples with consistency
        example = self_consistency(examples, anchor_text=anchor_text)
        consistency_examples.append(example)

        # get necessary info for evaluation
        trace = example['trace'] if 'trace' in example else []
        retrieval = (example['retrieval'] or []) if 'retrieval' in example else []
        retrieval = [r if len(r) == 2 else ('', r) for r in retrieval]  # add a empty query
        qid = example['qid'] if 'qid' in example else example['id']

        question = example['question'] if 'question' in example else None
        ref = choose_reference(example)
        final_ans = choose_final_answer(example)
        ans_id = example['answer_id'] if 'answer_id' in example else None
        pred = choose_full_prediction(example)
        if remove_followup:
            raw_pred = pred
            rms, rme = remove_followup
            pred = re.sub(f'{rms}[^\{rme}]*\{rme}', '', raw_pred)
            fu = ' '.join(re.findall(f'{rms}[^\{rme}]*\{rme}', raw_pred))
            followups.append(fu)
        probs = -np.log(example['output_prob']) if 'output_prob' in example else []
        final_metrics['avg_nll'] += np.mean(probs)
        final_metrics['ppl'] += np.sum(probs)
        final_metrics['tokens'] += len(probs)

        retrieval_ratios.append(len(retrieval) / (len(trace) or 1))
        num_rets.append(len(retrieval))
        rhit = len(set([r for (query, rs) in retrieval for r in rs if r.startswith(qid)]))
        retrieval_hits.append(rhit)

        references.append(ref)
        predictions.append(pred)
        num_steps.append(len(trace))

        if retrieval:
            ret_dids = np.array([rs if type(rs[0]) is str else rs[0] for (query, rs) in retrieval], dtype=np.str_)
        else:
            ret_dids = np.array([['placeholder']], dtype=np.str_)

        pred_ans = get_final_answer_from_pred(pred) if anchor_text else pred
        wrongformat = len(pred_ans) == 0
        if wrongformat:
            final_metrics['wrongformat'] += 1
        else:
            if dataset in {'strategyqa'}:
                correct = int(final_ans.lower() == pred_ans.lower())
                final_metrics['correct'] += correct
                final_metrics['incorrect'] += 1 - correct
            elif dataset in {'2wikihop'}:
                add_metric_kvs(WikiMultiHopQA.exact_match_score(pred_ans, final_ans, ground_truth_id=ans_id))
                add_metric_kvs(WikiMultiHopQA.f1_score(pred_ans, final_ans, ground_truth_id=ans_id))
            elif dataset in {'wikiasp'}:
                add_metric_kvs(WikiAsp.entity_f1_score(pred_ans, final_ans))
                print('qid', qid, WikiAsp.entity_f1_score(pred_ans, final_ans))
            elif dataset in {'asqa'}:
                print('qid', qid, ASQA.entity_f1_score(pred_ans, final_ans))
                add_metric_kvs(ASQA.entity_f1_score(pred_ans, final_ans))
            else:
                raise NotImplementedError

        has_search = '[Search(' in pred
        scount += has_search
        if has_search:
            search_per_example.append(len(re.findall('\[Search\(', pred)))

        if debug:
            print('ID->', qid)
            print('Q->', question)
            print()
            print('T->')
            for prompt, cont in trace:
                print(prompt)
                print('->', cont)
                print('\n------------------\n')
            print()
            print('P->', pred)
            print()
            print('G->', ref)
            input()

        # retrieval
        ret_accs.append([])
        ret_covers.append([])
        if ret_dids is not None:
            ret_seq_len = len(ret_dids)
            rel_dids: List[str] = np.array([d for d, r in qrels[qid].items() if r]) if qrels else []
            rels = np.isin(ret_dids, rel_dids).any(-1)  # (ret_seq_len)
            prev_pt = 0
            for pt in retrieval_percentiles:
                if type(pt) is int:
                    pass
                elif type(pt) is float:
                    pt = int(ret_seq_len * pt)
                else:
                    raise NotImplementedError
                if pt <= prev_pt:  # at least one token
                    pt = prev_pt + 1
                ret_accs[-1].append(rels[prev_pt:pt].mean())
                ret_covers[-1].append(len(np.intersect1d(ret_dids[:pt].reshape(-1), rel_dids)) / (len(rel_dids) or 1))
                prev_pt = max(min(pt, ret_seq_len - 1), 0)

    if root_file:
        with open(root_file + '.merge', 'w') as fout:
            for e in consistency_examples:
                fout.write(json.dumps(e) + '\n')

    total = len(predictions)  # change total

    format_list = lambda arr: ', '.join(map(lambda x: '{:.3f}'.format(x), arr.tolist()))

    # overall stats
    print('#pred\t#gold\t#examples')
    print(f'{np.mean([len(p) for p in predictions])}\t{np.mean([len(r) if type(r) is str else np.mean([len(_r) for _r in r]) for r in references])}\t{total}')
    print('')

    # retrieval-related stats
    ret_accs = np.array(ret_accs, dtype=float).mean(0)
    ret_covers = np.array(ret_covers, dtype=float).mean(0)
    print(f'#search\t#steps\tret ratio\tret hit')
    print(f'{np.mean(num_rets)}\t{np.mean(num_steps)}\t{np.mean(retrieval_ratios)}\t{np.mean(retrieval_hits)}')

    # major metrics
    print('\t'.join(final_metrics.keys()))
    print('\t'.join(map(lambda kv: str(np.exp(kv[1] / (final_metrics['tokens'] or 1))) if kv[0] == 'ppl' else str(kv[1] / total), final_metrics.items())))
    print('')

    # rouge metrics
    if dataset == 'lmdata':
        metrics = {}
    else:
        metrics = metric_func.compute(predictions=predictions, references=references)
    if remove_followup:
        metrics_followup = metric_func.compute(predictions=followups, references=references)
    print('\t'.join(metrics.keys()))
    print('\t'.join(map(str, metrics.values())))

    if remove_followup:
        print('\t'.join(metrics_followup.keys()))
        print('\t'.join(map(str, metrics_followup.values())))
        print('#pred\t#gold')
        print(f'{np.mean([len(p) for p in followups])}\t{np.mean([len(r) for r in references])}')
        print('')


def build_elasticsearch(
    beir_corpus_file_pattern: str,
    index_name: str,
):
    beir_corpus_files = glob.glob(beir_corpus_file_pattern)
    print(f'#files {len(beir_corpus_files)}')
    from beir.retrieval.search.lexical.elastic_search import ElasticSearch
    config = {
        'hostname': 'localhost',
        'index_name': index_name,
        'keys': {'title': 'title', 'body': 'txt'},
        'timeout': 100,
        'retry_on_timeout': True,
        'maxsize': 24,
        'number_of_shards': 'default',
        'language': 'english',
    }
    es = ElasticSearch(config)

    # create index
    print(f'create index {index_name}')
    es.delete_index()
    time.sleep(5)
    es.create_index()

    # generator
    def generate_actions():
        for beir_corpus_file in beir_corpus_files:
            with open(beir_corpus_file, 'r') as fin:
                reader = csv.reader(fin, delimiter='\t')
                header = next(reader)  # skip header
                for row in reader:
                    _id, text, title = row[0], row[1], row[2]
                    es_doc = {
                        '_id': _id,
                        '_op_type': 'index',
                        'refresh': 'wait_for',
                        config['keys']['title']: title,
                        config['keys']['body']: text,
                    }
                    yield es_doc

    # index
    progress = tqdm(unit='docs')
    es.bulk_add_to_index(
        generate_actions=generate_actions(),
        progress=progress)


def jsonl_to_keyvalue(
    jsonl_file: str,
    keyvalue_file: str,
    prefix_to_remove: List[str],
):
    with open(jsonl_file, 'r') as fin, open(keyvalue_file, 'w') as fout:
        key2output: Dict[str, str] = {}
        for l in fin:
            l = json.loads(l)
            pred = l['output'].strip()
            if prefix_to_remove:
                find = None
                for pattern in prefix_to_remove:
                    find = re.compile(pattern).search(pred)
                    if find:
                        pred = find.group(1)
                        break
                if find is None:
                    logging.warning(f'format error "{pred}"')
            key2output[l['qid']] = pred
        json.dump(key2output, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=[
        'eval', 'build_elasticsearch', 'jsonl_to_keyvalue'])
    parser.add_argument('--inp', type=str, default=None, nargs='+', help='input file')
    parser.add_argument('--dataset', type=str, default='2wikihop', help='input dataset', choices=[
        'strategyqa', '2wikihop', 'wikiasp', 'asqa'])
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0301', help='model name', choices=[
        'code-davinci-002', 'gpt-3.5-turbo-0301'])
    parser.add_argument('--out', type=str, default=None, help='output file')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    if args.task == 'eval':
        dataset = args.dataset
        jsonl_files = glob.glob(args.inp[0])
        if dataset == 'strategyqa':
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text=['answer is (yes|no)\.'],
                beir_dir='data/strategyqa/train_cot_beir')
        elif dataset == '2wikihop':
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text=['answer is (.*)'],
                beir_dir='data/2wikimultihopqa/dev_beir')
        elif dataset == 'wikiasp':
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                beir_dir=None)
        elif dataset in {'asqa'}:
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text=None,
                prefix_to_remove=[
                    'The answers to all interpretations are\: (.*)$',
                    'The answer to this interpretation is\: (.*)$',
                    'The answer to this interpretation is (.*)$'],
                beir_dir=None)

    elif args.task == 'build_elasticsearch':
        beir_corpus_file_pattern, index_name = args.inp  # 'wikipedia_dpr'
        build_elasticsearch(beir_corpus_file_pattern, index_name=index_name)

    elif args.task == 'jsonl_to_keyvalue':
        jsonl_file = args.inp[0]
        keyvalue_file = args.out
        jsonl_to_keyvalue(
            jsonl_file,
            keyvalue_file,
            prefix_to_remove=[
                'The answers to all interpretations are\: (.*)$',
                'The answer to this interpretation is\: (.*)$',
                'The answer to this interpretation is (.*)$'])
