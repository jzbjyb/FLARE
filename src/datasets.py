from typing import Dict, List, Callable, Tuple, Union, Callable
import logging
import os
import json
import re
import glob
import string
from collections import Counter
from urllib.parse import unquote
from tqdm import tqdm
import numpy as np
import spacy
from datasets import Dataset, concatenate_datasets, load_from_disk
from beir.datasets.data_loader import GenericDataLoader
logging.basicConfig(level=logging.INFO)


class BaseDataset:
    spacy_nlp = spacy.load('en_core_web_sm')
    use_model = 'spacy'
    nlp = eval(f'{use_model}_nlp')

    @classmethod
    def get_ner(cls, text):
        doc = cls.nlp(text)
        return list(doc.ents)

    @classmethod
    def entity_f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: str = None,
        debug: bool = False,
    ):
        if type(ground_truth) is str:
            ground_truth = [ground_truth]
        p = r = f1 = num_ent = 0
        for gold in ground_truth:
            pred_ents: List[str] = [cls.normalize_answer(ent.text) for ent in cls.get_ner(prediction)]
            gold_ents: List[str] = [cls.normalize_answer(ent.text) for ent in cls.get_ner(gold)]
            common_ents = Counter(pred_ents) & Counter(gold_ents)
            num_common_ents: int = sum(common_ents.values())
            if debug:
                print('PP', prediction)
                print('GG', gold)
                print('P', pred_ents)
                print('G', gold_ents)
                print('C', common_ents)
            _p = (num_common_ents / len(pred_ents)) if len(pred_ents) else 1
            _r = (num_common_ents / len(gold_ents)) if len(gold_ents) else 1
            assert _p <= 1 and _r <= 1
            _f1 = (2 * _p * _r) / ((_p + _r) or 1)
            p, r, f1 = max(p, _p), max(r, _r), max(f1, _f1)
            num_ent += len(gold_ents)
        num_ent /= len(ground_truth)
        return {'ent_f1': f1, 'ent_precision': p, 'ent_recall': r, 'num_ent': num_ent}

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(cls.get_all_alias(ground_truth_id))
        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)

            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def format(
        self,
        fewshot: int = 0,
    ):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            # ctx
            example['demo'] = demo
            example['case'] = case
            return example
        self.dataset = self.dataset.map(_format_for_dataset)

    def retrieval_augment_examplars(
        self,
        qagent: "QueryAgent",
        add_index: bool = False,
        use_gold: bool = False,
    ):
        return self.retrieval_augment_examplars_prepend(qagent, add_index=add_index, use_gold=use_gold)

    def retrieval_augment_examplars_prepend(
        self,
        qagent: "QueryAgent",
        add_index: bool = False,
        use_gold: Union[bool, Callable] = False,
    ):
        if use_gold:
            for examplar in tqdm(self.examplars, desc='ret aug demo'):
                _id = examplar['id']
                ctxs: List[Tuple[str, str]] = use_gold(_id)
                examplar['ctxs'] = ctxs
        else:  # search question
            qs = [examplar['question'] for examplar in self.examplars]
            ctx_ids, ctx_texts = qagent.retrieve(qs, is_question=True)  # (num_examplars, ret_topk) * 2
            for i in range(len(self.examplars)):
                self.examplars[i]['ctxs'] = list(zip(ctx_ids[i], ctx_texts[i]))


class StrategyQA(BaseDataset):
    cot_examplars: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctxs': [(None, "Hamsters are prey animals."),
                (None, "Prey animals provide food for predators.")],
            'cot': ('Hamsters are prey animals. ',
                'Prey are food for predators. ',
                'Thus, hamsters provide food for some animals.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctxs': [(None, "Brooke Shields graduated from Princeton University."),
                (None, "Princeton is ranked as the number 1 national college by US news."),
                (None, "University of Pennsylvania is ranked as number 6 national college by US news."),
                (None, "Princeton only admits around 6 percent of applicants as of 2018."),
                (None, "University of Pennsylvania accepts around 9% of applicants as of 2018.")],
            'cot': ('Brooke Shields went to Princeton University. ',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. ',
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctxs': [(None, "Hydrogen is the first element and has an atomic number of one."),
                (None, "The Spice Girls has five members."),
                (None, "To square a number, you multiply it by itself.")],
            'cot': ("Hydrogen has an atomic number of 1. ",
                "1 squared is 1. ",
                "There are 5 Spice Girls. ",
                "Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctxs': [(None, "Frost isn't uncommon to see during the month of December, as it is the winter."),
                (None, "College commencement ceremonies often happen during the months of December, May, and sometimes June.")],
            'cot': ("College commencement ceremonies can happen in December, May, and June. ",
                "December is in the winter, so there can be frost. ",
                "Thus, there could be frost at some commencements."),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctxs': [(None, "The War in Vietnam (1945-46) lasted around 6 months."),
                (None, "The gestation period for a llama is 11 months.")],
            'cot': ("The War in Vietnam was 6 months. ",
                "The gestation period for a llama is 11 months, which is more than 6 months. ",
                "Thus, a llama could not give birth twice during the War in Vietnam."),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctxs': [(None, "The density of a raw pear is about 0.59 g/cm^3."),
                (None, "The density of water is about 1 g/cm^3."),
                (None, "Objects only sink if they are denser than the surrounding fluid.")],
            'cot': ("The density of a pear is about 0.6g/cm^3, which is less than water. ",
                "Objects less dense than water float. ",
                "Thus, a pear would float."),
            'answer': 'no',
        }
    ]
    cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Generate a yes or no answer to the following question.\nQuestion: {ques}\nAnswer:'
    cot_output_template = lambda self, cot, ans: f'{cot} So the final answer is {ans}.'

    def __init__(self, beir_dir: str, prompt_type: str = 'cot'):
        assert prompt_type in {'cot'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        self.dataset = self.load_data(beir_dir)

    def load_data(self, beir_dir: str):
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='dev')
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                question = example['text']
                cot = example['metadata']['cot'] if 'cot' in example['metadata'] else ''
                ans = example['metadata']['answer']
                rel_dids = [did for did, rel in qrels[qid].items() if rel]
                ctxs = [(did, corpus[did]['text']) for did in rel_dids]
                output = self.output_template(cot, ans)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'cot': cot,
                    'answer': ans,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        return Dataset.from_list(dataset)


class WikiMultiHopQA(BaseDataset):
    wid2alias_file: str = 'data/2wikimultihopqa/dev_beir/id_aliases.json'

    cot_examplars: List[Dict] = [
        {
            'id': '5811079c0bdc11eba7f7acde48001122',
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
            'answer': "19 June 2013",
            "type": "compositional",
        },
        {
            'id': '35bf3490096d11ebbdafac1f6bf848b6',
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
            'answer': "no",
            "type": "comparison",
        },
        {
            'id': '97954d9408b011ebbd84ac1f6bf848b6',
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
            'answer': "no",
            "type": "bridge comparison",
        },
        {
            'id': 'cdbb82ec0baf11ebab90acde48001122',
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
            'answer': "Genghis Khan",
            "type": "inference",
        },

        {
            'id': 'c6805b2908a911ebbd80ac1f6bf848b6',
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
            'answer': "Martin Hodge",
            "type": "comparison",
        },
        {
            'id': 'e5150a5a0bda11eba7f7acde48001122',
            'question': "When did the director of film Laughter In Hell die?",
            'cot': "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
            'answer': "August 25, 1963",
            "type": "compositional",
        },
        {
            'id': 'a5995da508ab11ebbd82ac1f6bf848b6',
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
            'answer': "Twenty Plus Two",
            "type": "bridge comparison",
        },
        {
            'id': '1ceeab380baf11ebab90acde48001122',
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
            'answer': "Prithvipati Shah",
            "type": "inference",
        }
    ]
    cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    cot_output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    cot_ret_examplars = cot_examplars
    cot_ret_demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step): '
    cot_ret_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search):'
    cot_ret_output_template = cot_output_template

    sa_examplars: List[Dict] = [
        {
            'id': '5811079c0bdc11eba7f7acde48001122',
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who directed the film Hypocrite?\n"
                "Intermediate answer: Miguel Morayta.\n"
                "Follow up: When did Miguel Morayta die?\n"
                "Intermediate answer: 19 June 2013."
            ),
            'answer': "19 June 2013",
            "type": "compositional",
        },
        {
            'id': '35bf3490096d11ebbdafac1f6bf848b6',
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: In which country is Kurram Garhi located?\n"
                "Intermediate answer: Pakistan.\n"
                "Follow up: In which country is Trojkrsti located?\n"
                "Intermediate answer: Republic of Macedonia."
            ),
            'answer': "no",
            "type": "comparison",
        },
        {
            'id': '97954d9408b011ebbd84ac1f6bf848b6',
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who directed the film Coolie No. 1 (1995 Film)?\n"
                "Intermediate answer: David Dhawan.\n"
                "Follow up: Who directed the film The Sensational Trial?\n"
                "Intermediate answer: Karl Freund.\n"
                "Follow up: What is the nationality of David Dhawan?\n"
                "Intermediate answer: India.\n"
                "Follow up: What is the nationality of Karl Freund?\n"
                "Intermediate answer: Germany."
            ),
            'answer': "no",
            "type": "bridge comparison",
        },
        {
            'id': 'cdbb82ec0baf11ebab90acde48001122',
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who is the spouse of Boraqchin?\n"
                "Intermediate answer: Ögedei Khan.\n"
                "Follow up: Who is the father of Ögedei Khan?\n"
                "Intermediate answer: Genghis Khan."
            ),
            'answer': "Genghis Khan",
            "type": "inference",
        },

        {
            'id': 'c6805b2908a911ebbd80ac1f6bf848b6',
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: When was Martin Hodge born?\n"
                "Intermediate answer: 4 February 1959.\n"
                "Follow up: When was Ivania Martinich born?\n"
                "Intermediate answer: 25 July 1995."
            ),
            'answer': "Martin Hodge",
            "type": "comparison",
        },
        {
            'id': 'e5150a5a0bda11eba7f7acde48001122',
            'question': "When did the director of film Laughter In Hell die?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who directed the film Laughter In Hell?\n"
                "Intermediate answer: Edward L. Cahn.\n"
                "Follow up: When did Edward L. Cahn die?\n"
                "Intermediate answer: August 25, 1963."
            ),
            'answer': "August 25, 1963",
            "type": "compositional",
        },
        {
            'id': 'a5995da508ab11ebbd82ac1f6bf848b6',
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who directed the film Twenty Plus Two?\n"
                "Intermediate answer: Joseph M. Newman.\n"
                "Follow up: Who directed the film The Gal Who Took the West?\n"
                "Intermediate answer: Frederick de Cordova.\n"
                "Follow up: When did Joseph M. Newman die?\n"
                "Intermediate answer: January 23, 2006.\n"
                "Follow up: When did Fred de Cordova die?\n"
                "Intermediate answer: September 15, 2001."
            ),
            'answer': "Twenty Plus Two",
            "type": "bridge comparison",
        },
        {
            'id': '1ceeab380baf11ebab90acde48001122',
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who is the child of Krishna Shah?\n"
                "Intermediate answer: Rudra Shah.\n"
                "Follow up: Who is the child of Rudra Shah?\n"
                "Intermediate answer: Prithvipati Shah."
            ),
            'answer': "Prithvipati Shah",
            "type": "inference",
        },
    ]
    sa_demo_input_template = sa_test_input_template = lambda self, ques: f'Question: {ques}\n'
    sa_output_template = lambda self, cot, ans: f'{cot}\nSo the final answer is {ans}.'

    def __init__(self, beir_dir: str, prompt_type: str = 'cot'):
        self.wid2alias_file = os.path.join(beir_dir, 'id_aliases.json')
        assert prompt_type in {'cot', 'cot_ret', 'sa'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        self.dataset = self.load_data(beir_dir)

    @classmethod
    def load_wid2alias(cls):
        if hasattr(cls, 'wid2alias'):
            return
        cls.wid2alias: Dict[str, List[str]] = {}
        with open(cls.wid2alias_file, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                cls.wid2alias[l['Q_id']] = l['aliases']

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        cls.load_wid2alias()
        if ground_truth_id and ground_truth_id in cls.wid2alias:
            return cls.wid2alias[ground_truth_id]
        return []

    def load_data(self, beir_dir: str):
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                question = example['text']
                ans = example['metadata']['answer']
                ans_id = example['metadata']['answer_id']
                ctxs = example['metadata']['ctxs']
                output = self.output_template(cot='', ans=ans)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'answer_id': ans_id,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        return Dataset.from_list(dataset)


class WikiAsp(BaseDataset):
    title_annotation_file: str = 'data/wikiasp/wikiasp_titles_annotated.tsv'
    cot_examplars: List[Dict] = [
        {
            "id": "train-45-14962",
            "question": "Aslanhane Mosque including the following aspects: location, history",
            "answer_raw": "# location\nthe mosque is in the old quarter of ankara next to ankara castle . with an altitude of 947 metres ( 3 , 107  ft ) it overlooks ankara at 39 \u00b0 56 \u2032 12 \u2033 n 32 \u00b0 51 \u2032 55 \u2033 e .\n# history\nthe mosque is one of the oldest mosques in turkey still standing . it was built during the reign of mesud ii of the anatolian seljuks in 1290 . its architect was ebubekir mehmet . it was commissioned by two ahi leaders named h\u00fcsamettin and hasaneddin . however , in 1330 , it was repaired by another ahi leader named \u015ferafettin after whom the mosque was named . after several minor repairs the mosque was restored by the directorate general of foundations in 2010 - 2013 term .",
            "answer": "# Location\nThe mosque is in the old quarter of ankara next to ankara castle. With an altitude of 947 metres (3,107 ft) it overlooks ankara at 39°56′12″N 32°51′55″E.\n# History\nThe mosque is one of the oldest mosques in Turkey still standing. It was built during the reign of Mesud II of the Anatolian Seljuks in 1290. Its architect was Ebubekir Mehmet. It was commissioned by two Ahi leaders named Hüsamettin and Hasaneddin. However, in 1330, it was repaired by another Ahi leader named Şerafettin after whom the mosque was named. After several minor repairs the mosque was restored by the directorate general of foundations in 2010-2013 term.",
            "domain": "building"
        },
        {
            "id": "train-64-5493",
            "question": "Untold Legends: The Warrior's Code including the following aspects: reception, gameplay, development",
            "answer_raw": "# reception\nthe game received \" mixed or average reviews \" according to video game review aggregator metacritic .\n# gameplay\nthe warrior ' s code is a hack n ' slash action role - playing game , which concentrates on action - oriented combat .\n# development\nas a pre - order bonus , the game was shipped with a small action figure of the guardian class .",
            "answer": "# Reception\nThe game received \"mixed or average reviews\" according to video game review aggregator Metacritic.\n# Gameplay\nThe warrior's code is a hack n' slash action role-playing game, which concentrates on action-oriented combat.\n# Development\nAs a pre-order bonus, the game was shipped with a small action figure of the Guardian class.",
            "domain": "software"
        },
        {
            "id": "train-39-10878",
            "question": "Raid on St. Augustine including the following aspects: aftermath, background",
            "answer_raw": "# aftermath\nonce the english had gone men\u00e9ndez and the rest of the spanish settlers returned to find a smoldering ruins and very little left . he soon and begged for help from the viceroy of cuba and the settlement took a while to build itself back up . the destroyed fort was replaced with the present day castillo de san marcos .\n# background\nwar had already been unofficially declared by philip ii of spain after the treaty of nonsuch in which elizabeth i had offered her support to the rebellious protestant dutch rebels . the queen through francis walsingham ordered sir francis drake to lead an expedition to attack the spanish new world in a kind of preemptive strike . sailing from plymouth , england , he struck first at santiago in november 1585 then across the atlantic at the spanish new world city of santo domingo of which was captured and ransomed on 1 january 1586 and following that successfully attacked the important city of cartagena on 19 february . drake wanted to strike at another spanish city on the main before finally visiting and replenishing sir walter raleigh ' s new colony of roanoke colony on the american east coast . then after this he hoped to make the transatlantic crossing back to england . the fleet headed north , and in late april drake put into the spanish cuban mainland and his men dug wells in search of fresh water and gathered supplies to help counter an outbreak of dysentery after which he moved on . the fleet traveled north within sight of land on the florida peninsula sailing past the west coast . on 27 may 1586 as they approached further north a small fort was spotted on the shore , with a small inlet close by . this was the location of st augustine , the most northerly town in spain ' s new world empire , and the oldest permanent colonial settlement in north america . drake knew of the place and was also aware of the fact that the spanish under pedro men\u00e9ndez de avil\u00e9s had ordered all of the french huguenot colonists that had tried to settle in the area executed . drake decided on one final opportunity to raid and plunder , and a chance to avenge his fellow protestants .",
            "answer": "# Aftermath\nOnce the English had gone Menéndez and the rest of the Spanish settlers returned to find a smoldering ruins and very little left. He soon and begged for help from the viceroy of Cuba and the settlement took a while to build itself back up. The destroyed fort was replaced with the present day Castillo de San Marcos.\n# Background\nWar had already been unofficially declared by Philip II of Spain after the Treaty of Nonsuch in which Elizabeth I had offered her support to the rebellious Protestant Dutch rebels. The Queen through Francis Walsingham ordered Sir Francis Drake to lead an expedition to attack the Spanish New World in a kind of preemptive strike. Sailing from Plymouth, England, he struck first at Santiago in November 1585 then across the Atlantic at the Spanish new world city of Santo Domingo of which was captured and ransomed on 1 January 1586 and following that successfully attacked the important city of Cartagena on 19 February. Drake wanted to strike at another Spanish city on the Main before finally visiting and replenishing Sir Walter Raleigh's new colony of Roanoke Colony on the American East Coast. Then after this he hoped to make the Transatlantic crossing back to England. The fleet headed north, and in late April Drake put into the Spanish Cuban mainland and his men dug wells in search of fresh water and gathered supplies to help counter an outbreak of dysentery after which he moved on. The fleet traveled north within sight of land on the Florida peninsula sailing past the West coast. On 27 May 1586 as they approached further north a small fort was spotted on the shore, with a small inlet close by. This was the location of St Augustine, the most northerly town in Spain's New World Empire, and the oldest permanent colonial settlement in North America. Drake knew of the place and was also aware of the fact that the spanish under Pedro Menéndez de Avilés had ordered all of the French Huguenot colonists that had tried to settle in the area executed. Drake decided on one final opportunity to raid and plunder, and a chance to avenge his fellow Protestants.",
            "domain": "event"
        },
        {
            "id": "train-53-18686",
            "question": "Lakewood (Livingston, Alabama) including the following aspects: architecture, history",
            "answer_raw": "# architecture\nthe house has a plan that is relatively rare in early alabama architecture . the plan features a brick ground floor that is topped by one - and - a - half - stories of wood - frame construction . the ground floor originally contained domestic spaces , with the formal rooms on the principle floor and bedrooms on the upper floor . a central hallway is present on all levels . the facade is five bays wide , with central entrance doors on the ground and principle floors . the bays are divided by two - story doric pilasters , with the middle third of the facade occupied by a two - tiered tetrastyle doric portico . two curved wrought iron staircases ascend from ground level to the front center of the upper portico , leading to the formal entrance .\n# history\nlakewood was built for joseph lake , a native of north carolina , by hiram w . bardwell , a master builder . construction was completed in 1840 . located adjacent to the university of west alabama , julia strudwick tutwiler , a lake relative , periodically resided in the house from 1881 to 1910 while she served as president of the university . it was then known as livingston normal college . the house was extensively photographed by alex bush for the historic american buildings survey in november and december 1936 . lakewood has continued to be owned by descendants of the lake family to the current day . the house and its surviving 10 acres ( 4 . 0  ha ) of grounds were listed on the places in peril in 2012 due to the immediate threat of its acquisition by developers .",
            "answer": "# Architecture\nThe house has a plan that is relatively rare in early Alabama architecture. The plan features a brick ground floor that is topped by one-and-a-half-stories of wood-frame construction. The ground floor originally contained domestic spaces, with the formal rooms on the principle floor and bedrooms on the upper floor. A central hallway is present on all levels. The facade is five bays wide, with central entrance doors on the ground and principle floors. The bays are divided by two-story Doric pilasters, with the middle third of the facade occupied by a two-tiered tetrastyle Doric portico. Two curved wrought iron staircases ascend from ground level to the front center of the upper portico, leading to the formal entrance.\n# History\nLakewood was built for Joseph lake, a native of North Carolina, by Hiram W. Bardwell, a master builder. Construction was completed in 1840. Located adjacent to the University of West Alabama, Julia Strudwick Tutwiler, a Lake relative, periodically resided in the house from 1881 to 1910 while she served as president of the university. It was then known as Livingston Normal College. The house was extensively photographed by Alex Bush for the Historic American Buildings Survey in November and December 1936. Lakewood has continued to be owned by descendants of the Lake family to the current day. The house and its surviving 10 acres (4.0 ha) of grounds were listed on the Places in Peril in 2012 due to the immediate threat of its acquisition by developers.",
            "domain": "historic_place"
        },
        {
            "id": "train-50-1349",
            "question": "Echo School (Oregon) including the following aspects: academics, history",
            "answer_raw": "# academics\nin 2008 , 91 % of the school ' s seniors received their high school diploma . of 66 students , 60 graduated , 1 dropped out , 3 received a modified diploma , and 2 were still in high school in 2009 .\n# history\nthe class of 2008 was the 100th class in the school ' s history .",
            "answer": "# Academics\nIn 2008, 91% of the school's seniors received their high school diploma. Of 66 students, 60 graduated, 1 dropped out, 3 received a modified diploma, and 2 were still in high school in 2009.\n# History\nThe class of 2008 was the 100th class in the school's history.",
            "domain": "educational_institution"
        },
        {
            "id": "train-73-14144",
            "question": "Melaleuca serpentina including the following aspects: taxonomy and naming, description, distribution and habitat",
            "answer_raw": "# taxonomy and naming\nmelaleuca serpentina was first formally described in 2009 by lyndley craven in novon from a specimen collected adjacent to the woodsreef asbestos mine near barraba . in 2012 , udovicic and spencer gave the species the name callistemon serpentinus but in 2013 , craven transferred all species previously known as callistemon to melaleuca . some authorities continue to use callistemon serpentinus . the specific epithet ( serpentina ) refers to this species often occurring on soils derived from serpentinite . callistemon serpentinus is regarded as a synonym of melaleuca serpentina by the royal botanic gardens , kew .\n# description\nmelaleuca serpentina is a shrub growing to 4  m ( 10  ft ) tall with hard , papery bark . its leaves are arranged alternately and are 21 \u2013 53  mm ( 0 . 8 \u2013 2  in ) long , 2 \u2013 5  mm ( 0 . 08 \u2013 0 . 2  in ) wide , more or less flat , narrow elliptical to egg - shaped with the narrow end towards the base and an end tapering to a sharp point . the leaves have a mid - vein but the lateral veins are obscure and there are many distinct oil glands . the flowers are creamy green to yellow and are arranged in spikes on the ends of branches which continue to grow after flowering and also in the leaf axils . the spikes are 30 \u2013 40  mm ( 1 \u2013 2  in ) in diameter with 15 to 35 individual flowers . the petals are 2 . 2 \u2013 4  mm ( 0 . 09 \u2013 0 . 2  in ) long and fall off as the flower ages and there are 37 to 51 stamens in each flower . flowering occurs in april , october and december and is followed by fruit which are woody capsules , 4 . 2 \u2013 4 . 6  mm ( 0 . 17 \u2013 0 . 18  in ) long .\n# distribution and habitat\nmelaleuca serpentina occurs in the barraba district growing in grassy woodland on soils derived from serpentinite .",
            "answer": "# Taxonomy and naming\nMelaleuca serpentina was first formally described in 2009 by Lyndley Craven in Novon from a specimen collected adjacent to the Woodsreef asbestos mine near Barraba. In 2012 , Udovicic and Spencer gave the species the name Callistemon serpentinus but in 2013, Craven transferred all species previously known as Callistemon to Melaleuca. Some authorities continue to use Callistemon serpentinus. The specific epithet (serpentina) refers to this species often occurring on soils derived from serpentinite. Callistemon serpentinus is regarded as a synonym of Melaleuca serpentina by the Royal Botanic Gardens, Kew.\n# Description\nMelaleuca serpentina is a shrub growing to 4 m (10 ft) tall with hard, papery bark. Its leaves are arranged alternately and are 21–53 mm (0.8–2 in) long, 2–5 mm (0.08–0.2 in) wide, more or less flat, narrow elliptical to egg-shaped with the narrow end towards the base and an end tapering to a sharp point. The leaves have a mid-vein but the lateral veins are obscure and there are many distinct oil glands. The flowers are creamy green to yellow and are arranged in spikes on the ends of branches which continue to grow after flowering and also in the leaf axils. The spikes are 30–40 mm (1–2 in) in diameter with 15 to 35 individual flowers. The petals are 2.2–4 mm (0.09–0.2 in) long and fall off as the flower ages and there are 37 to 51 stamens in each flower. flowering occurs in April, October and December and is followed by fruit which are woody capsules, 4.2–4.6 mm (0.17–0.18 in) long.\n# Distribution and habitat\nMelaleuca serpentina occurs in the Barraba district growing in grassy woodland on soils derived from serpentinite.",
            "domain": "plant"
        },
        {
            "id": "train-62-1235",
            "question": "The Making of the Mob including the following aspects: reception, production",
            "answer_raw": "# reception\nthe first season received mixed responses from television critics and a metacritic score of 59 out of 100 , based on six reviews , indicating \" mixed or average reviews \" . the review aggregator website rotten tomatoes reported a 40 % \" rotten \" critics rating based on five reviews .\n# production\non january 10 , 2015 , amc ordered the series as a \" special event \" miniseries to air in mid - 2015 . on july 31 , 2015 , two weeks after the series premiere , amc renewed it for a second season to air in mid - 2016 .",
            "answer": "# Reception\nThe first season received mixed responses from television critics and a Metacritic score of 59 out of 100, based on six reviews, indicating \"mixed or average reviews\". The review aggregator website Rotten Tomatoes reported a 40% \"rotten\" critics rating based on five reviews.\n# Production\nOn January 10, 2015, AMC ordered the series as a \"special event\" miniseries to air in mid-2015. On July 31, 2015, two weeks after the series premiere, AMC renewed it for a second season to air in mid-2016.",
            "domain": "television_show"
        },
        {
            "id": "train-35-4670",
            "question": "Green Township, Scioto County, Ohio including the following aspects: geography, name and history, government",
            "answer_raw": "# geography\nlocated in the far south of the county along the ohio river , it borders the following townships : porter township - north vernon township - northeast elizabeth township , lawrence county - east hamilton township , lawrence county - southeast greenup county , kentucky lies across the ohio river to the west . no municipalities are located in green township , although the census - designated place of franklin furnace lies in the northeastern part of the township , and the unincorporated community of haverhill lies in the south of the township . both of these communities are ohio river towns .\n# name and history\nnamed after griffin green , a land agent , it is one of sixteen green townships statewide . origins of green township date to between 1803 and 1811 . the community of haverhill was settled as early as 1797 . the powellsville community dates to july 31 , 1846 .\n# government\nthe township is governed by a three - member board of trustees , who are elected in november of odd - numbered years to a four - year term beginning on the following january 1 . two are elected in the year after the presidential election and one is elected in the year before it . there is also an elected township fiscal officer , who serves a four - year term beginning on april 1 of the year after the election , which is held in november of the year before the presidential election . vacancies in the fiscal officership or on the board of trustees are filled by the remaining trustees .",
            "answer": "# Geography\nLocated in the far south of the county along the Ohio River, it borders the following townships: Porter Township - north Vernon Township - northeast Elizabeth Township, Lawrence County - east Hamilton Township, Lawrence County - southeast Greenup County, Kentucky lies across the Ohio River to the west. No municipalities are located in Green Township, although the census-designated place of Franklin Furnace lies in the northeastern part of the township, and the unincorporated community of Haverhill lies in the south of the township. Both of these communities are Ohio River towns.\n# Name and history\nNamed after Griffin Green, a land agent, it is one of sixteen Green Townships statewide. Origins of Green Township date to between 1803 and 1811. The community of Haverhill was settled as early as 1797. the Powellsville community dates to July 31, 1846.\n# Government\nThe township is governed by a three-member board of trustees, who are elected in November of odd-numbered years to a four-year term beginning on the following January 1. Two are elected in the year after the presidential election and one is elected in the year before it. There is also an elected township fiscal officer, who serves a four-year term beginning on April 1 of the year after the election, which is held in November of the year before the presidential election. Vacancies in the fiscal officership or on the board of trustees are filled by the remaining trustees.",
            "domain": "town"
        }
    ]  # shuffled
    cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Generate a summary about {ques} with one aspect per line.\n'
    cot_output_template = lambda self, cot, ans: ans

    def __init__(self, hf_dataset_dir_pattern: str, prompt_type: str = 'cot'):
        assert prompt_type in {'cot'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        self.dataset = self.load_data(hf_dataset_dir_pattern)

    @staticmethod
    def get_wiki_url(urls: List[str]):
        for url in urls:
            if 'wikipedia.org' in url:
                return url
        return None

    @staticmethod
    def wiki_url_to_title(url: str):
        title = ' '.join(unquote(url).rsplit('/wiki/', 1)[-1].split('_')) if url else url
        return title

    @classmethod
    def load_id2title(cls):  # load id2title that is human annotated to corret machine generated title
        cls.id2title = {}
        if os.path.exists(cls.title_annotation_file):
            cls.id2title = dict([tuple(l.strip().split('\t')) for l in open(cls.title_annotation_file)])

    def load_data(self, hf_dataset_dir_pattern: str):
        self.load_id2title()

        def map_fn(example):
            qid = example['exid']
            title = example['clean_title'] if 'clean_title' in example else example['title']
            if qid in self.id2title:
                logging.info(f'modify title based on annotation for {qid}')
                title = self.id2title[qid]
            references = ' '.join(example['inputs'])
            targets = example['clean_targets'] if 'clean_targets' in example else example['targets']
            aspects: List[str] = []
            summary: List[str] = []
            for asp, text in targets:
                asp, text = asp.strip(), text.strip().replace('\n', ' ')
                if len(text) <= 0:  # remove empty aspects
                    continue
                aspects.append(asp)
                summary.append(f'# {asp.capitalize()}\n{text}')
            summary: str = '\n'.join(summary)
            output = self.output_template(cot=None, ans=summary)
            question = f'{title} including the following aspects: {", ".join(aspects)}'
            new_example = {
                'qid': qid,
                'question': question,
                'raw_references': references,
                'answer': summary,
                'gold_output': output,
            }
            return new_example

        all_hf_dirs = glob.glob(hf_dataset_dir_pattern.strip('"'))  # TODO: fix this bug
        logging.info(f'loading from {all_hf_dirs}')
        data = concatenate_datasets([load_from_disk(hf_dir) for hf_dir in all_hf_dirs])
        return data.map(map_fn)


class ASQA(BaseDataset):
    general_hint_jsonl_file = 'data/asqa/ASQA_test_general_hint.jsonl'
    general_hint_in_input_examplars: List[Dict] = [
        {
            "id": "-6681997980074150658",
            "question": "Who played bonnie in gone with the wind?",
            "category": "entity",
            "hint_me": "This question is ambiguous because Gone with the Wind refers to multiple entities.",
            "general_hint": "This question is ambiguous in terms of which version or adaptation of Gone with the Wind is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify which version of Gone with the Wind is being referred to, and therefore could be interpreted as asking about either the 1939 film or the 2008 musical.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific film adaptation, as it could refer to either the 1939 film or the 2008 musical adaptation.",
            "subq_cot": "we need to consider different versions or adaptations of Gone with the Wind. Gone with the Wind has two versions or adaptations: the 1939 film Gone with the Wind or the 2008 musical Gone with the Wind.",
            "answer": "Therefore, this question has 2 interpretations: (1) Who played Bonnie in the 1939 film Gone with the Wind? (2) Who played Bonnie in the 2008 musical Gone with the Wind? The answers to all interpretations are: (1) The 1939 film Gone with the Wind\'s character Bonnie was played by Eleanore Cammack \"Cammie\" King. (2) The 2008 musical Gone with the Wind\'s character Bonnie was played by Leilah de Meza.",
        },
        {
            "id": "-1170854568854448296",
            "question": "What is the second largest city in the usa?",
            "category": "event",
            "hint_me": "This question is ambiguous because city size can be measured in multiple ways.",
            "general_hint": "This question is ambiguous in terms of the criteria being used to determine the second largest city in the USA.",
            "specific_hint": "This question is ambiguous as it does not specify whether the second largest city in the USA is being referred to by population or by area, and therefore could be interpreted as asking about either aspect.",
            "specific_hint_keyword": "This question is ambiguous in terms of the criteria used to determine the second largest city in the USA, as it could refer to either population or area.",
            "subq_cot": "we need to consider different criteria to determine a city's size. City size can be measured by two criteria: population or area.",
            "answer": "Therefore, this question has 2 interpretations: (1) What is the second largest city in the USA by population? (2) What is the second largest city in the USA by area? The answers to all interpretations are: (1) The second largest city in the USA by population is Los Angeles, California. (2) The second largest city in the USA by area is Juneau, Alaska.",
        },
        {
            "id": "-42361505900466516",
            "question": "When was bohemian rhapsody released as a single?",
            "category": "context",
            "hint_me": "This question is ambiguous because it has different answers in different countries.",
            "general_hint": "This question is ambiguous in terms of which country's release of the single is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify in which country Bohemian Rhapsody was released as a single, and therefore could be interpreted as asking about either the United Kingdom or the United States.",
            "specific_hint_keyword": "This question is ambiguous in terms of the geographic location of the release, as it could refer to either the United Kingdom or the United States.",
            "subq_cot": "we need to consider different countries where Bohemian Rhapsody is released. Bohemian Rhapsody was released in the United Kingdom and in the United States on different dates.",
            "answer": "Therefore, this question has 2 interpretations: (1) When was Bohemian Rhapsody released as a single in the United Kingdom? (2) When was Bohemian Rhapsody released as a single in the United States? The answers to all interpretations are: (1) Bohemian Rhapsody was released as a single in the United Kingdom on 31 October 1975. (2) Bohemian Rhapsody was released as a single in the United States on December 1975."
        },
        {
            "id": "-6158441934367575013",
            "question": "Where do the philadelphia eagles play their home games?",
            "category": "answer_type",
            "hint_me": "This question is ambiguous because there are multiple interpretations of the home field of the Philadelphia Eagles.",
            "general_hint": "This question is ambiguous in terms of which specific location or venue is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify which aspect of the Philadelphia Eagles' home games is being referred to, and therefore could be interpreted as asking about the city, sports complex, or stadium where they play their home games.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific location of the Philadelphia Eagles' home games, as it could refer to the city, sports complex, or stadium.",
            "subq_cot": "we need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium.",
            "answer": "Therefore, this question has 3 interpretations: (1) What city do the Philadelphia Eagles play their home games? (2) In what sports complex do the Philadelphia Eagles play their home games? (3) What stadium do the Philadelphia Eagles play their home games? The answers to all interpretations are: (1) Philadelphia Eagles play their home games in the city Philadelphia. (2) Philadelphia Eagles play their home games in the South Philadelphia Sports Complex. (3) Philadelphia Eagles play their home games in the Lincoln Financial Field stadium.",
        },

        {
            "id": "7925778961305870115",
            "question": "When did xbox one come out in australia?",
            "category": "entity",
            "hint_me": "This question is ambiguous because Xbox One refers to multiple entities.",
            "general_hint": "This question is ambiguous in terms of which specific version of the Xbox One is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify which version of the Xbox One is being referred to, and therefore could be interpreted as asking about either the original Xbox One or the Xbox One X.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific Xbox One model release, as it could refer to either the original Xbox One or the Xbox One X.",
            "subq_cot": "we need to consider the different versions of the Xbox One that have been released. Xbox One has two versions: the Xbox One video game console or the Xbox One X high-end model.",
            "answer": "Therefore, this question has 2 interpretations: (1) When did the Xbox One release in Australia? (2) When did the Xbox One X release in Australia? The answers to all interpretations are: (1) The Xbox One video game console was released in Australia on November 22, 2013. (2) The Xbox One X video game console was released in Australia on November 7, 2017.",
        },
        {
            "id": "-5527347701597533393",
            "question": "When does the movie summer of 84 come out?",
            "category": "event",
            "hint_me": "This question is ambiguous because a movie might come out on different dates depending on the context.",
            "general_hint": "This question is ambiguous in terms of which release of the movie is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify which release of the movie Summer of '84 is being referred to, and therefore could be interpreted as asking about either its release at the Sundance Festival or its release throughout the US.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific release of the movie Summer of '84, as it could refer to either the release date at the Sundance Festival or the release date throughout the US.",
            "subq_cot": "we need to consider different releases of the movie Summer of '84. The movie Summer of '84 is first released at the Sundance Festival before it's released throughout the US.",
            "answer": "Therefore, this question has 2 interpretations: (1) When did the movie Summer of '84 first release at the Sundance Festival? (2) When did the movie Summer of '84 first release throughout the US? The answers to all interpretations are: (1) Summer of '84 was released at the Sundance Festival on January 22, 2018. (2) Summer of '84 was released throughout the US on August 10, 2018.",
        },
        {
            "id": "8423232783444896189",
            "question": "What was roy orbison's first number one hit?",
            "category": "context",
            "hint_me": "This question is ambiguous because it has different answers in different countries.",
            "general_hint": "This question is ambiguous in terms of which specific chart or region is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify in which countries or regions Roy Orbison's first number one hit is being referred to, and therefore could be interpreted as asking about either the US Hot 100 and Canada or the UK and Ireland.",
            "specific_hint_keyword": "This question is ambiguous in terms of the geographic location of the chart where Roy Orbison's first number one hit is being referred to, as it could refer to either the US Hot 100 and Canada or the UK and Ireland.",
            "subq_cot": "we need to consider the different charts and regions where Roy Orbison's music was popular. Roy Orbison is popular in both the US Hot 100 and Canada, and the UK and Ireland.",
            "answer": "Therefore, this question has 2 interpretations: (1) What was Roy Orbison's first number one hit in the US Hot 100 and Canada? (2) What was Roy Orbison's first number one hit in the UK and Ireland? The answers to all interpretations are: (1) Running Scared was the first number one hit for Roy Orbison in the US Hot 100 and Canada. (2) Only the Lonely (Know the Way I Feel) was the first number one hit for Roy Orbison in the UK and Ireland.",
        },
        {
            "id": "3471060247311635100",
            "question": "What is the criminal's name in the breakfast club?",
            "category": "answer_type",
            "hint_me": "This question is ambiguous because there are multiple interpretations of the criminal's name.",
            "general_hint": "This question is ambiguous in terms of which specific name is being referred to - the character's name or the actor's name.",
            "specific_hint": "This question is ambiguous as it does not specify which aspect of the criminal in The Breakfast Club is being referred to, and therefore could be interpreted as asking about either the character's name or the actor's name who played the character.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific identity of the criminal in The Breakfast Club, as it could refer to either the character name or the actor who played the role.",
            "subq_cot": "we need to consider both possibilities: the character's name or the actor's name.",
            "answer": "Therefore, this question has 2 interpretations: (1) What is the criminal's character name in The Breakfast Club? (2) What is the the name of the actor who played the criminal in The Breakfast Club? The answers to all interpretations are: (1) John Bender was the name of the criminal's character in The Breakfast Club. (2) Judd Nelson was the actor of the criminal in The Breakfast Club.",
        },


        {
            "id": "-6497998034447212269",
            "question": "When did bat out of hell come out?",
            "category": "entity",
            "hint_me": "This question is ambiguous because Bat out of Hell refers to multiple entities.",
            "general_hint": "This question is ambiguous in terms of which specific version or adaptation of Bat Out of Hell is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify which version of Bat Out of Hell is being referred to, and therefore could be interpreted as asking about either the album or the TV series.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific media format of Bat Out of Hell, as it could refer to either the album or the TV series.",
            "subq_cot": "we need to consider the different versions or adaptations of Bat Out of Hell. Bat Out of Hell has two versions or adaptations: the album Bat Out of Hell or the TV series Bat Out of Hell.",
            "answer": "Therefore, this question has 2 interpretations: (1) When did the album Bat Out of Hell come out? (2) When did the TV series Bat Out of Hell come out? The answers to all interpretations are: (1) The album Bat Out of Hell came out on October 21, 1977. (2) The British television show Bat Out of Hell came out on 26 November 1966.",
        },
        {
            "id": "4370113190341229231",
            "question": "When was smoking banned in new york city?",
            "category": "event",
            "hint_me": "This question is ambiguous because smoking ban in NYC happened progressively and it has multiple interpretations.",
            "general_hint": "This question is ambiguous in terms of which specific smoking ban in New York City is being referred to.",
            "specific_hint": "This question is ambiguous as it does not specify which aspect of smoking ban in New York City is being referred to, and therefore could be interpreted as asking about the ban on indoor smoking, the statewide smoking ban, the ban on smoking in parks and rec centers, or the ban on smoking for anyone under 21.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific smoking ban being referred to, as it could refer to the ban on indoor smoking, the statewide smoking ban, the ban on smoking in parks and rec centers, or the ban on smoking for anyone under 21 in NYC.",
            "subq_cot": "we need to consider the different smoking bans that have been implemented in New York City. Smoking ban in NYC has multiple implementations: indoor smoking ban, statewide smoking ban, smoking ban in parks and rec centers, or smoking ban for anyone under 21.",
            "answer": "Therefore, this question has 4 interpretations: (1) When was indoor smoking banned in NYC? (2) When did New Yorks statewide smoking ban go into effect? (3) When was smoking in parks and rec centers banned in NYC? (4) When was anyone under 21 banned from smoking in NYC? The answers to all interpretations are: (1) Indoor smoking in NYC was banned on March 30, 2003. (2) New York went to a state wide ban on July 24, 2003. (3) Smoking was banned in NYC parks and rec centers on May 23, 2011. (4) NYC banned smoking for anyone under the age of 21 on May 18, 2014.",
        },
        {
            "id": "-4377718773044986307",
            "question": "New zealand is a part of what continent?",
            "category": "context",
            "hint_me": "This question is ambiguous because it has different answers in different history period.",
            "general_hint": "This question is ambiguous in terms of whether it is asking about the current or historical continental location of New Zealand.",
            "specific_hint": "This question is ambiguous as it does not specify which aspect of New Zealand's continental history is being referred to, and therefore could be interpreted as asking about either its current microcontinent or its past supercontinent before the Jurassic period.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific geographic context being referred to, as it could refer to the microcontinent that New Zealand is a part of or the supercontinent that New Zealand was a part of until the Jurassic period.",
            "subq_cot": "we need to consider both possibilities: current or historical continental location. The contient of New Zealand is different before and after Jurassic period.",
            "answer": "Therefore, this question has 2 interpretations: (1) New Zealand is a part of what microcontienent? (2) New Zealand was a part of what supercontinent until the Jurassic period? The answers to all interpretations are: (1) New Zealand is currently part of a continent called Zealandia. (2) New Zealand was a part of Gondwana until the Jurassic period.",
        },
        {
            "id": "8905159142292415847",
            "question": "Who sings i stand alone in quest for camelot?",
            "category": "answer_type",
            "hint_me": "This question is ambiguous because there are multiple interpretations of the singer.",
            "general_hint": "This question is ambiguous in terms of which specific type of performer is being referred to - the character or the artist.",
            "specific_hint": "This question is ambiguous as it does not specify which aspect of the song \"I Stand Alone\" in Quest for Camelot is being referred to, and therefore could be interpreted as asking about either the character who sings the song or the artist who performs the song.",
            "specific_hint_keyword": "This question is ambiguous in terms of the specific identity of the singer of I Stand Alone in Quest for Camelot, as it could refer to either the character or the artist who performed the song.",
            "subq_cot": "we need to consider both possibilities: the character or the artist.",
            "answer": "Therefore, this question has 2 interpretations: (1) Which character sings I Stand Alone in Quest for Camelot? (2) Which artist sings I Stand Alone in Quest for Camelot? The answers to all interpretations are: (1) The character sings I Stand Alone in Quest for Camelot is King Arthur. (2) The artist sings I Stand Alone in Quest for Camelot is Steve Perry.",
        }
    ]
    general_hint_in_input_output_template = lambda self, cot, ans: ans
    general_hint_in_input_demo_input_template = general_hint_in_input_test_input_template = lambda self, ques: f'Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.\nQuestion: {ques}\nAnswer: In order to figure out its interpretations,'

    general_hint_in_output_examplars = general_hint_in_input_examplars
    general_hint_in_output_output_template = general_hint_in_input_output_template
    general_hint_in_output_test_input_template = general_hint_in_output_demo_input_template = lambda self, ques: f'Given an ambiguous question, figure out its interpretations and answer them one by one.\nQuestion: {ques}\nAnswer:'

    def __init__(self, json_file: str = None, split: str = 'dev', prompt_type: str = 'cot'):
        assert prompt_type in {'general_hint_in_input', 'general_hint_in_output'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        self.dataset = self.load_data(json_file, split, prompt_type=prompt_type)
        if prompt_type == 'general_hint_in_input':
            for e in self.examplars:
                e['question'] = f'{e["question"]}\nHint: {e["general_hint"]}'
                e['answer'] = f'{e["subq_cot"]} {e["answer"]}'
        elif prompt_type == 'general_hint_in_output':
            for e in self.examplars:
                e['answer'] = f'{e["general_hint"]} In order to figure out its interpretations, {e["subq_cot"]} {e["answer"]}'

    def load_data(self, json_file: str = None, split: str = 'dev', prompt_type: str = None):
        return self._load_data(json_file=json_file, split=split, prompt_type=prompt_type, output_template=self.output_template)

    @classmethod
    def _load_data(cls, json_file: str = None, split: str = 'dev', prompt_type: str = None, output_template: Callable = None):
        def clean_hint(hint: str):
            from_start = 'The original question is ambiguous'
            to_start = 'This question is ambiguous'
            hint = hint.strip()
            if hint.startswith(from_start):
                hint = to_start + hint[len(from_start):]
            return hint

        qid2genhint: Dict[str, str] = {}
        if cls.general_hint_jsonl_file and os.path.exists(cls.general_hint_jsonl_file):
            for l in open(cls.general_hint_jsonl_file):
                l = json.loads(l)
                qid2genhint[l['qid']] = clean_hint(l['output'])

        dataset = []
        num_hasctx = 0
        with open(json_file, 'r') as fin:
            data = json.load(fin)[split]
            for key, example in data.items():
                qid = key
                question = example['ambiguous_question']
                sub_questions: List[str] = []
                answers: List[str] = []
                title2content: Dict[str, str] = {}
                for ann in example['annotations']:
                    ans = ann['long_answer'].strip()
                    answers.append(ans)
                    for know in ann['knowledge']:
                        title2content[know['wikipage']] = know['content']
                for qa in example['qa_pairs']:
                    sub_questions.append(qa['question'].strip())
                    if qa['wikipage'] is None:
                        continue
                    title2content[qa['wikipage']] = qa['context']
                assert len(answers) >= 1
                assert len(sub_questions) >= 1
                answers = sorted(answers, key=lambda x: -len(x))  # sort based on length
                output = output_template(cot=None, ans=answers[0])
                ctxs: List[Tuple[str, str]] = list(title2content.items())  # could be empty
                num_hasctx += int(len(ctxs) > 0)

                hint = None
                if prompt_type == 'general_hint_in_input':
                    hint = qid2genhint[qid] if qid in qid2genhint else None
                    question = f'{question}\nHint: {hint}'
                elif prompt_type == 'general_hint_in_output':
                    hint = qid2genhint[qid] if qid in qid2genhint else None

                dataset.append({
                    'qid': qid,
                    'question': question,
                    'sub_questions': sub_questions,
                    'hint': hint,
                    'answer': answers[0],
                    'answers': answers,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        logging.info(f'{num_hasctx} / {len(dataset)} have gold ctxs')
        return Dataset.from_list(dataset)
