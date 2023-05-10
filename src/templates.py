from typing import List, Dict, Any, Tuple, Union
from operator import itemgetter
import copy
from collections import namedtuple
import spacy
from nltk.tokenize.punkt import PunktSentenceTokenizer
import tiktoken
from .utils import openai_api_call, Utils


class CtxPrompt:
    ctx_position: str = 'begin'
    ret_instruction: "RetrievalInstruction" = None
    instruction: str = None
    format_reference_method: str = 'default'
    clean_reference: bool = False
    add_ref_suffix: str = None
    add_ref_prefix: str = None

    def __init__(
        self,
        demo: List["CtxPrompt"] = [],
        ctx: str = None,
        ctxs: List[Tuple[str, str]] = [],
        case: str = None,
        question: str = None,
        qid: str = None,
        gold_output: str = None,
    ):
        assert self.ctx_position in {'before_case', 'begin'}
        self.demo = demo
        self.did = None
        self.ctx = ctx
        self.ctxs = ctxs
        self._ctxs = []  # used for ctx alwayed being used
        self.ctxs_idx = 0
        self.case = case
        self.question = question or case
        self.qid = qid
        self.gold_output = gold_output
        self.ind = 1  # ctx index
        self.gen_len = 0
        self.gold_used_len = 0

    @classmethod
    def from_dict(cls, adict):
        adict = dict(adict)
        if 'demo' in adict:
            adict['demo'] = [cls.from_dict(d) for d in adict['demo']]
        return cls(**{k: adict[k] for k in ['demo', 'ctx', 'ctxs', 'case', 'question', 'qid', 'gold_output'] if k in adict})

    @classmethod
    def clean_rets(cls, rets: List[str]) -> List[str]:
        return [ret.replace('\n', ' ').strip() for ret in rets if ret.replace('\n', ' ').strip()]

    @classmethod
    def chatgpt_get_response(cls, prompt: Union[str, List[str]], examplars: List[List[Tuple[str, str]]] = [[]], max_tokens: int = 2048, api_key: str = None):
        is_single = type(prompt) is str
        if is_single:
            prompt = [prompt]
            examplars = examplars or [[]]
        if len(prompt) != len(examplars):
            examplars = [[] for _ in range(len(prompt))]
        for p in prompt:
            assert len(p.split()) <= max_tokens
        responses = openai_api_call(
            api_key=api_key,
            model='gpt-3.5-turbo-0301',
            messages=[[
                {'role': 'user' if i == 0 else 'assistant', 'content': e[i]} for e in es for i in range(2)
            ] + [
                {'role': 'user', 'content': p},
            ] for p, es in zip(prompt, examplars)],
            temperature=0.0,
            top_p=0.0,
            max_tokens=max_tokens)
        generations = [r['choices'][0]['message']['content'] for r in responses]
        if is_single:
            assert len(generations) == 1
            return generations[0]
        return generations

    @classmethod
    def canonicalize_text(cls, text: Union[str, List[str]], field: str = 'paragraph', api_key: str = None, debug: bool = False):
        is_single = type(text) is not list
        if is_single:
            text = [text]
        prompts = [f'For the following {field}, remove unnecessary spaces and capitalize words properly.\n{field.capitalize()}:\n{t}' for t in text]
        clean_texts = cls.chatgpt_get_response(prompts, api_key=api_key)
        post_clean_texts = []
        for ct, t in zip(clean_texts, text):
            if ct.strip().startswith(f'Sorry, there is no {field} provided'):
                post_clean_texts.append(t)
            else:
                post_clean_texts.append(ct)
        if debug:
            for p, ct in zip(prompts, post_clean_texts):
                print('-' * 10)
                print(p)
                print('-' * 10)
                print(ct)
                print('-' * 10)
            input()
        if is_single:
            assert len(post_clean_texts) == 1
            return post_clean_texts[0]
        return post_clean_texts

    @classmethod
    def annotate_low_confidence_terms(cls, tokens: List[str], probs: List[float], low: float = 0.0, special_symbol: str = '*', min_gap: int = 5):
        # mark with symbol
        text = []
        prev_is_low = -1
        has = False
        for i, (token, prob) in enumerate(zip(tokens, probs)):
            if prob <= low:
                if prev_is_low == -1 or i - prev_is_low >= min_gap:
                    has = True
                    leading_spaces = len(token) - len(token.lstrip())
                    if leading_spaces <= 0:
                        text.append(f'*{token}')
                    else:
                        text.append(f'{token[:leading_spaces]}*{token[leading_spaces:]}')
                    prev_is_low = i
                else:
                    text.append(token)
            else:
                text.append(token)
        text = ''.join(text)
        return text, has

    @classmethod
    def extract_low_confidence_terms_rule(
        cls,
        tokens: List[str],
        probs: List[float],
        low: float = 0.0,
        min_gap: int = 5,  # TODO: the minimal token-based gap to separate two terms
        expand: bool = True,
        exclude_punct: bool = True,
        always_extract_low: bool = False,
        api_key: str = None):
        prev_low_pos = -1
        has = False
        terms: List[List[str]] = []
        spans: List[Tuple[int, int]] = []
        for i, (token, prob) in enumerate(zip(tokens, probs)):
            if prob <= low:
                if prev_low_pos == -1 or i - prev_low_pos >= min_gap:
                    # new term
                    terms.append([token])
                    spans.append((i, i + 1))
                else:
                    # old term
                    for j in range(prev_low_pos + 1, i + 1):
                        terms[-1].append(tokens[j])
                    spans[-1] = (spans[-1][0], i + 1)
                prev_low_pos = i
        terms = [''.join(term).strip() for term in terms]
        if len(spans) <= 0:
            return terms
        if expand:
            new_terms = cls.extract_constituents(tokens, spans=spans, api_key=api_key)
            assert len(new_terms) == len(terms)
            if always_extract_low:
                terms = [nt if nt is not None else t for nt, t in zip(new_terms, terms)]
            else:
                terms = [nt for nt in new_terms if nt is not None]
        if exclude_punct:
            terms = [t for t in terms if t not in Utils.punctuations]
        return terms

    @classmethod
    def extract_constituents(cls, tokens: List[str], spans: List[Tuple[int, int]], api_key: str = None, special_symbol: str = '*', debug: bool = False):
        examplars = [
            ("Egypt has one of the longest histories of any country, tracing its heritage along *the Nile Delta back to the 6th–4th millennia BCE.", "*the Nile", "the Nile Delta"),
            ("The settlement, which legal experts said was the largest struck by an American media company, was announced by the two sides and the judge in the case at the *11th hour.", "*11th", "11th hour"),
            ("In his only surviving love letter to her, written a few months before their wedding, Tyler promised, \"*Whether I float or sink in the stream of fortune, you may be assured of this, that I shall never cease to love you.\"", "*Whether I float", "Whether I float or sink in the stream of fortune")
        ]
        prompt_format = lambda sent, term: f"{sent}\n\nGiven the above sentence, extract the term/entity/phrase starting with \"{term}\"."

        # add special_symbol
        ori_sent = ''.join(tokens)
        cases: List[Tuple[str, str]] = []
        for start_ind, end_ind in spans:
            start_token = tokens[start_ind]
            n_lead_spaces = len(start_token) - len(start_token.lstrip())
            if n_lead_spaces <= 0:
                tokens[start_ind] = f'*{start_token}'
            else:
                tokens[start_ind] = f'{start_token[:n_lead_spaces]}*{start_token[n_lead_spaces:]}'
            sent = ''.join(tokens).strip()
            term = ''.join(tokens[start_ind:end_ind]).strip()
            cases.append((sent, term))
            tokens[start_ind] = start_token  # convert tokens back to the original state

        # call
        prompts: List[str] = [prompt_format(s, t) for s, t in cases]
        examplars: List[Tuple[str, str]] = [(prompt_format(s, t), out) for s, t, out in examplars]
        responses = cls.chatgpt_get_response(prompt=prompts, examplars=[examplars] * len(prompts), api_key=api_key)

        # post-process
        constituents: List[str] = []
        for r, (sent, term), prompt in zip(responses, cases, prompts):
            if term.startswith(special_symbol):  # trim special_symbol
                term = term[len(special_symbol):].strip()
            if debug:
                print('-' * 10)
                print(prompt)
                print('-' * 10)
                print(r)
                print('-' * 10)
            r = r.strip().split('\n', 1)[0].strip()
            if r.startswith(special_symbol):  # trim special_symbol
                r = r[len(special_symbol):].strip()
            if not r.startswith(term):  # not an expansion
                r = None
            elif r not in ori_sent:  # skip non-existent terms
                r = None
            elif not r:  # empty
                r = None
            constituents.append(r)
        return constituents

    @classmethod
    def extract_low_confidence_terms(cls, context: str, tokens: List[str], probs: List[float], low: float = 0.0, api_key: str = None, special_symbol: str = '*', debug: bool = False):
        examplars = [
            ('*Egypt has one of the longest histories of any country, tracing its heritage along *the Nile Delta back to the *6th–4th millennia BCE.', '*Egypt\n*the Nile Delta\n*6th–4th'),
            ('The settlement, which *legal experts said was *the largest struck by an American media company, was *announced by the two sides and the judge in the case at the 11th hour.', '*legal experts\n*the largest struck\n*announced'),
            ('In his only *surviving love letter to her, written a few months before their wedding, Tyler promised, "*Whether I *float or sink in the stream of fortune, you may be assured of this, that I shall never *cease to love you."', '*surviving love letter\n*Whether\n*float or sink\n*cease to love you')
        ]
        original_text = ''.join(tokens)
        text, has = cls.annotate_low_confidence_terms(tokens=tokens, probs=probs, low=low, special_symbol=special_symbol)
        if not has:
            return []
        # extract terms
        prompt_format = lambda x: f'Given the following sentence, extract all terms/entities starting with the symbol "{special_symbol}", one at a line.\n{x}'
        examplars = [(prompt_format(inp), out) for inp, out in examplars]
        prompt = prompt_format(text)
        response = cls.chatgpt_get_response(prompt, examplars=examplars, api_key=api_key)
        terms = [t.strip() for t in response.strip().split('\n') if t.strip().startswith(special_symbol)]  # remove outlier
        terms = [t.lstrip(special_symbol) for t in terms if t in text and t.lstrip(special_symbol) in original_text]  # remove non-exist terms
        if debug:
            print('-' * 10)
            print(prompt)
            print('-' * 10)
            print(response)
            print('-' * 10)
            print(terms)
            print('-' * 10)
        return terms

    @classmethod
    def replace_low_confidence_terms(cls, context: str, tokens: List[str], probs: List[float], low: float = 0.0, api_key: str = None, special_symbol: str = '*', replace_symbol: str = 'XXX', debug: bool = False):
        text, has = cls.annotate_low_confidence_terms(tokens=tokens, probs=probs, low=low, special_symbol=special_symbol)
        if not has:
            return text
        # replace terms
        prompt = f'Given the previous context and the last sentence, detect all terms/entities in the last sentence starting with the symbol "{special_symbol}", then replace them with "{replace_symbol}".\nPrevious context:\n{context}\nLast sentence:\n{text}'
        replaced_text = cls.chatgpt_get_response(prompt, api_key=api_key)
        if debug:
            print('-' * 10)
            print(prompt)
            print('-' * 10)
            print(replaced_text)
            print('-' * 10)
        return replaced_text

    @classmethod
    def replace_low_confidence_terms_by_extract(cls, context: str, tokens: List[str], probs: List[float], low: float = 0.0, api_key: str = None, special_symbol: str = '*', replace_symbol: str = 'XXX', min_term_length: int = 0):
        text = ''.join(tokens)
        terms = cls.extract_low_confidence_terms(context=context, tokens=tokens, probs=probs, low=low, api_key=api_key, special_symbol=special_symbol)
        for term in terms:
            if min_term_length and len(term) <= min_term_length:  # ignore short terms
                continue
            text = text.replace(term, replace_symbol)
        return text

    @classmethod
    def decontextualize_text(cls, context: str, text: str, api_key: str = None, debug: bool = False):
        examplars = []
        start_sym, end_sym = "=== Text (start) ===", "=== Text (start) ==="
        prompt_format = lambda x, y: f'Replace pronouns in the following text with their corresponding references.\n\n{x.strip()}\n{start_sym}\n{y.strip()}\n{end_sym}'
        examplars = [(prompt_format(e[0], e[1]), e[2]) for e in examplars]
        prompt = prompt_format(context, text)
        decontext_text = cls.chatgpt_get_response(prompt, examplars=examplars, api_key=api_key).strip()
        decontext_text = decontext_text.split(start_sym, 1)[-1].strip()
        decontext_text = decontext_text[:-len(end_sym)] if decontext_text.endswith(end_sym) else decontext_text
        if debug:
            print('-' * 10)
            print(prompt)
            print('-' * 10)
            print(decontext_text)
            print('-' * 10)
        return decontext_text

    @classmethod
    def ask_question_text(
        cls,
        context: str,
        text: str,
        terms: List[str],
        api_key: str = None,
        debug: bool = False,
        filter_question: bool = True,
        ask_full_text: bool = False,
        use_full_text: bool = True,
    ):
        questions: List[str] = []
        cases: List[str] = []
        for term in terms:
            term = term.strip('"')
            case = f'{context.lstrip()}{text.rstrip()}\n\nGiven the above passage, ask a question to which the answer is the term/entity/phrase "{term}".'
            cases.append(case)
        if ask_full_text and len(terms) <= 0:
            case = f'{context.lstrip()}{text.rstrip()}\n\nGiven the above passage, ask a question to which the answer is the information contained in the last sentence "{text.strip()}".'
            cases.append(case)
        elif use_full_text and len(terms) <= 0:
            return [text.strip()]

        responses = cls.chatgpt_get_response(cases, api_key=api_key)

        questions: List[str] = []
        for case, question in zip(cases, responses):
            question = question.strip()
            if filter_question and not question.endswith('?'):
                continue
            questions.append(question)
            if debug:
                print('-' * 10)
                print(case)
                print('-' * 10)
                print(question)
                print('-' * 10)
        return questions

    @classmethod
    def get_queries_from_text_for_retrieval(
        cls,
        context: str,
        tokens: List[str],
        probs: List[float],
        low: float = 0.0,
        api_key: str = None,
        replace_symbol: str = 'XXX',
        detect_low_terms: bool = False,
        decontextualize: bool = False,
        askquestion: bool = False,
        debug: bool = False,
    ) -> List[str]:
        text = ''.join(tokens)
        if debug:
            print('0->', context)
            print('1->', text)
            print(list(zip(tokens, probs)))
        if detect_low_terms:
            terms = cls.extract_low_confidence_terms_rule(tokens=tokens, probs=probs, low=low, api_key=api_key)
        if debug:
            print('2->', terms)
        if decontextualize:
            text = cls.decontextualize_text(context=context, text=text, api_key=api_key)
            if debug:
                print('3->', text)
        elif askquestion:
            questions = cls.ask_question_text(context=context, text=text, terms=terms, api_key=api_key)
        if detect_low_terms:
            if decontextualize:
                for term in terms:
                    questions = [text.replace(term, ' ')]
            elif askquestion:
                pass
        if debug:
            print('4->', questions)
            input()
        return questions

    def get_query_for_retrieval(self):
        if self.gen_len == 0:
            return self.question
        else:
            return self.case

    def get_all_ctxs(self) -> List[str]:
        return self.ctxs

    def add_generation(self, cont: str):
        self.case += cont
        self.gen_len += len(cont)
        if self.gold_used_len != 0:  # use gold
            self.gold_output = self.gold_output[self.gold_used_len:]
            self.gold_used_len = 0

    def reset_generation(self):
        if self.gen_len <= 0:
            return
        self.case = self.case[:-self.gen_len]
        self.gen_len = 0

    def change_ctx(self):
        assert len(self.ctxs)
        if self.ctxs_idx >= len(self.ctxs):
            return self.did, self.ctx
        self.did, self.ctx = self.ctxs[self.ctxs_idx]
        self.ctxs_idx += 1
        return self.did, self.ctx

    def reinit_ctx(self):
        self.ctx = None
        self.ind = 1

    def check_ctx(self, method):
        if self.ctx:
            return
        if self._ctxs:
            self.update_retrieval([], method=method)

    def update_retrieval(
        self,
        rets: List[Tuple[str, str]] = [],
        method: str = 'replace',
        dedup: bool = True,
        add_index: bool = True,
    ):
        if self._ctxs:  # merge with kept ctxs
            exist_ids = set([_id for _id, t in self._ctxs])
            new_rets = copy.deepcopy(self._ctxs)
            for _id, t in rets:
                if _id not in exist_ids:
                    new_rets.append((_id, t))
                    exist_ids.add(_id)
            rets = new_rets
        rets = list(map(itemgetter(1), rets))
        rets = self.clean_rets(rets)
        def merge_rets():
            if add_index:
                return '\n'.join(f'[{self.ind + i}]: {ret}' for i, ret in enumerate(rets))
            return '\n'.join(rets)
        assert method in {'replace', 'append'}
        merge_ret = merge_rets()
        if self.ctx is None:
            self.ctx = merge_ret
        else:
            if method == 'replace':
                self.ctx = merge_ret
            elif method == 'append':
                if dedup:
                    if merge_ret.lower() not in self.ctx.lower():
                        self.ctx += '\n' + merge_ret
                        self.ind += len(rets)
                else:
                    self.ctx += '\n' + merge_ret
                    self.ind += len(rets)
            else:
                raise NotImplementedError

    @classmethod
    def format_reference(cls, ref: str, api_key: str = None):
        if cls.add_ref_suffix and not ref.endswith(cls.add_ref_suffix):
            ref += cls.add_ref_suffix
        if cls.add_ref_prefix and not ref.startswith(cls.add_ref_prefix):
            ref = cls.add_ref_prefix + ref
        if cls.clean_reference:
            ref = cls.canonicalize_text(ref, field='text', api_key=api_key)
        method = cls.format_reference_method
        assert method in {'default', 'searchresults', 'searchresultsrank'}
        if method == 'default':
            return 'Reference: ' + ref
        if method == 'searchresults':
            return 'Search results :\n' + ref
        if method == 'searchresultsrank':
            return 'Search results ranked based on relevance in descending order:\n' + ref
        raise NotImplementedError

    def get_prefix(
            self,
            qagent: "QueryAgent",
            prefix_method: str = 'sentence') -> Tuple[str, int]:
        if not self.gold_output:  # finish
            return qagent.final_stop_sym, 0
        if prefix_method == 'sentence':
            prefix, self.gold_used_len = ApiReturn.get_sent(self.gold_output, position='begin')
            return prefix, 0
        elif prefix_method == 'all':
            prefix, self.gold_used_len = self.gold_output, len(self.gold_output)
            return prefix, 0
        elif prefix_method.startswith('sentence_first:'):
            firstk = int(prefix_method[len('sentence_first:'):])
            prefix, self.gold_used_len = ApiReturn.get_sent(self.gold_output, position='begin')
            prefix = qagent.get_tokens(prefix, topk=firstk)[0]
            return prefix, None
        elif prefix_method.startswith('freq:'):
            firstk = int(prefix_method[len('freq:'):])
            prefix, self.gold_used_len = qagent.get_tokens(self.gold_output, topk=firstk)
            return prefix, 0
        else:
            raise NotImplementedError

    def format(
        self,
        use_ctx: bool = False,
        use_ret_instruction: bool = True,
        use_instruction: bool = True,
        is_chat_model: bool = False,
        api_key: str = None
    ):
        # run on demo
        demo_formatted: List[str] = [d.format(use_ctx=use_ctx, use_ret_instruction=False, use_instruction=False)[0] for d in self.demo]

        use_ctx = use_ctx and bool(self.ctx)  # do not use ctx when it's None or empty string
        use_ret_instruction = use_ret_instruction and self.ret_instruction is not None
        ref = self.format_reference(self.ctx, api_key=api_key) if use_ctx else None
        task, ret, ensemble = self.ret_instruction.format(use_ctx=use_ctx) if use_ret_instruction else (None, None, None)
        elements: List[str] = []

        if use_ctx and self.ctx_position == 'begin':
            elements.append(ref)

        # append retrieval instructionj
        if use_ret_instruction:
            elements.append(ret)

        # append task instruction
        if use_ret_instruction:
            elements.append(task)

        # append additional instruction
        if use_instruction and self.instruction is not None:
            elements.append(self.instruction)

        # append demo
        if len(demo_formatted) and not is_chat_model:
            elements.extend(demo_formatted)

        # append ensemble
        if use_ret_instruction:
            elements.append(ensemble)

        if use_ctx and self.ctx_position == 'before_case':
            elements.append(ref + '\n' + self.case)
        else:
            elements.append(self.case)

        return '\n\n'.join(elements), self.gen_len, demo_formatted


Sentence = namedtuple('Sentence', 'text start_char end_char')


class ApiReturn:
    EOS = '<|endoftext|>'
    spacy_nlp = spacy.load('en_core_web_sm')
    psentencizer = PunktSentenceTokenizer()
    use_sentencizer = 'nltk'
    min_sent_len = 5

    def __init__(
        self,
        prompt: str,
        text: str,
        tokens: List[str] = None,
        probs: List[float] = None,
        offsets: List[int] = None,
        finish_reason: str = 'stop',
        model: str = None,
        skip_len: int = 0,
    ):
        self.model = model
        self.prompt = prompt
        self.text = text

        self.tokens = tokens
        self.probs = probs
        self.offsets = offsets
        if self.has_tokens:
            assert len(tokens) == len(probs) == len(offsets)

        self.finish_reason = finish_reason
        if self.finish_reason is None:
            self.finish_reason = 'stop'  # TODO: a bug from openai?

        if skip_len:  # skip `skip_len` chars at the beginning
            self.text = self.text[skip_len:]
            if self.has_tokens:
                i = 0
                for i, off in enumerate(self.offsets):
                    if off == skip_len:
                        break
                    elif off > skip_len:  # the previous token span across the boundary
                        i = i - 1
                        assert i >= 0
                        break
                self.tokens = self.tokens[i:]
                self.probs = self.probs[i:]
                self.offsets = self.offsets[i:]

    @property
    def has_tokens(self):
        return self.tokens is not None

    @property
    def token_probs(self):
        if self.has_tokens:
            return self.probs
        else:
            return []

    @property
    def num_tokens(self):
        if self.has_tokens:
            return len(self.tokens)
        else:
            return len(tiktoken.encoding_for_model(self.model).encode(self.text))

    @property
    def has_endoftext(self):
        return self.EOS in self.tokens

    @property
    def is_empty(self):
        return len(self.text.strip()) == 0

    @classmethod
    def get_sent(cls, text: str, position: str = 'begin'):
        if cls.use_sentencizer == 'spacy':
            sents = list(cls.spacy_nlp(text).sents)
        elif cls.use_sentencizer == 'nltk':
            sents = [Sentence(text[s:e], s, e) for s, e in cls.psentencizer.span_tokenize(text)]
        else:
            raise NotImplementedError
        if position == 'begin':
            break_at = len(text)
            for sent in sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= cls.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break
            return text[:break_at], break_at
        if position == 'end':
            break_at = 0
            for i in range(len(sents)):
                sent = sents[len(sents) - i - 1]
                if len(text) - sent.start_char >= cls.min_sent_len:  # TODO: argument
                    break_at = sent.start_char
                    break
            return text[break_at:], break_at
        raise NotImplementedError

    def truncate_at_prob(self, low: float):
        assert self.has_tokens, 'not supported'

        if self.num_tokens <= 1:
            return self

        break_point = self.num_tokens
        for i in range(self.num_tokens):
            t, p, o = self.tokens[i], self.probs[i], self.offsets[i]
            if p <= low:
                break_point = i
                break
        if break_point == 0 and self.num_tokens > 0:  # avoid deadlock
            break_point = 1

        while break_point < self.num_tokens:  # truncation
            assert break_point > 0
            keep = self.offsets[break_point] - len(self.prompt)
            if keep <= 0:
                break_point += 1
                continue

            self.text = self.text[:keep]
            self.tokens = self.tokens[:break_point]
            self.probs = self.probs[:break_point]
            self.offsets = self.offsets[:break_point]
            self.finish_reason = 'boundary'
            break

        return self

    def truncate_at_boundary(self, unit: str = 'sentence'):
        if self.num_tokens <= 1:
            return self

        if unit == 'sentence':
            if self.use_sentencizer == 'spacy':
                sents = list(self.spacy_nlp(self.text).sents)
            elif self.use_sentencizer == 'nltk':
                sents = [Sentence(self.text[s:e], s, e) for s, e in self.psentencizer.span_tokenize(self.text)]
            else:
                raise NotImplementedError
            break_at = len(self.text)
            for sent in sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= self.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break

            if break_at > 0 and break_at < len(self.text):  # truncation
                if self.has_tokens:
                    i = 0
                    for i in range(self.num_tokens):
                        if self.offsets[i] - len(self.prompt) >= break_at:
                            break_at = self.offsets[i] - len(self.prompt)
                            break
                    assert i > 0
                    self.tokens = self.tokens[:i]
                    self.probs = self.probs[:i]
                    self.offsets = self.offsets[:i]
                assert break_at > 0
                self.text = self.text[:break_at]
                self.finish_reason = 'boundary'
        else:
            raise NotImplementedError
        return self

    def truncate_at_substring(self, substr: str):
        position = self.text.find(substr)
        if position == -1:
            return
        self.text = self.text[:position]
        if self.has_tokens:
            i = 0
            for i, off in enumerate(self.offsets):
                if off - len(self.prompt) == position:
                    break
                elif off - len(self.prompt) > position:  # the previous token span across the boundary
                    i = i - 1
                    assert i >= 0
                    break
            self.tokens = self.tokens[:i]
            self.probs = self.probs[:i]
            self.offsets = self.offsets[:i]

    def use_as_query(
        self,
        low_prob: float = None,
        mask_prob: float = None,
        mask_method: str = 'simple',
        n_gen_char_in_prompt: int = 0,
        api_key: str = None,
    ):
        if not low_prob and not mask_prob:
            return self.text
        assert self.has_tokens, 'not supported'
        if low_prob:
            ok = False
            for p in self.probs:
                if p <= low_prob:
                    ok = True
                    break
            if not ok:
                return ''
        if mask_prob:
            if mask_method == 'simple':
                keep = [(t if p > mask_prob else ' ') for t, p in zip(self.tokens, self.probs)]
                keep = ''.join(keep).strip()
                return keep
            elif mask_method in {'wholeterm-decontextualize', 'wholeterm-askquestion'}:
                if n_gen_char_in_prompt == 0:
                    context = ''
                else:
                    context = self.prompt[-n_gen_char_in_prompt:]
                decontextualize = 'decontextualize' in mask_method
                askquestion = 'askquestion' in mask_method
                keep = CtxPrompt.get_queries_from_text_for_retrieval(
                    context=context,
                    tokens=self.tokens,
                    probs=self.probs,
                    low=mask_prob,
                    api_key=api_key,
                    detect_low_terms=True,
                    decontextualize=decontextualize,
                    askquestion=askquestion)
                return keep
            else:
                raise NotImplementedError
        else:
            return self.text


class RetrievalInstruction:
    cot_instruction: Dict[str, Any] = {
        'retrieval': 'Skill 1. Use the Search API to look up relevant information by writing "[Search(term)]" where "term" is the search term you want to look up. For example:',
        'task': 'Skill 2. Answer questions by thinking step-by-step. First, write out the reasoning steps, then draw the conclusion. For example:',
        'ensemble': 'Now, combine the aforementioned two skills. First, write out the reasoning steps, then draw the conclusion, where the reasoning steps should also utilize the Search API "[Search(term)]" whenever possible.',
        'examplars': [
            {
                'question': 'But what are the risks during production of nanomaterials?',
                'ctxs': [(None, 'The increased production of manufactured nanomaterials (MNMs) and their use in consumer and industrial products means that workers in all countries will be at the front line of any exposure, placing...')],
                'answer': '[Search(nanomaterial production risks)] Some nanomaterials may give rise to various kinds of lung damage.',
            },
            {
                'question': 'The colors on the flag of Ghana have the following meanings.',
                'ctxs': [(None, "The flag of Ghana comprises of the Pan-African colors of red, yellow and green. These colors are horizontal stripes that make up the background of the flag. Red is represents the nation's fight for independence, the gold is a sign of the country's mineral wealth, and the green is a representation of the country's natural wealth...")],
                'answer': 'Red is for [Search(Ghana flag red meaning)] the blood of martyrs, green for forests, and gold for mineral wealth.',
            },
            {
                'question': 'Metformin is the first-line drug for what?',
                'ctxs': [(None, "Metformin, sold under the brand name Glucophage, among others, is the main first-line medication for the treatment of type 2 diabetes,[6][7][8][9] particularly in people who are overweight.[7] It is also used in the treatment of polycystic ovary syndrome...")],
                'answer': '[Search(Metformin first-line drug)] patients with type 2 diabetes and obesity.'
            }
        ]
    }

    strategyqa_instruction: Dict[str, Any] = {
        'task': 'Skill 2. Answer questions by thinking step-by-step. First, write out the reasoning steps, then generate a yes or no answer. For example:',
        'ensemble': 'Now, combine the aforementioned two skills. First, write out the reasoning steps, then generate a yes or no answer, where the reasoning steps should also utilize the Search API "[Search(term)]" whenever possible.',
    }

    summary_instruction: Dict[str, Any] = {
        'task': '2. You should generate a short paragraph of summary for an entity. For example:',
        'ensemble': '3. Now, you should combine the aforementioned two abilities. You should generate a short paragraph of summary for an entity and utilize the Search API "[Search(term)]" whenever possible.',
    }

    def __init__(self, method: str = 'cot', fewshot: int = None):
        self.instruction = getattr(self, f'{method}_instruction')
        for k, v in self.cot_instruction.items():
            if k not in self.instruction:
                self.instruction[k] = v
        self.fewshot = len(self.instruction['examplars']) if fewshot is None else self.fewshot

    def format(self, use_ctx: bool = False) -> Tuple[str, str]:
        use_ctx = False  # no ctx for examplars
        demos: List[str] = []
        for i in range(self.fewshot):
            q = self.instruction['examplars'][i]['question']
            a = self.instruction['examplars'][i]['answer']
            if use_ctx:
                ctxs = self.instruction['examplars'][i]['ctxs']
                assert CtxPrompt.ctx_position == 'before_case'
                ref = CtxPrompt.format_reference(' '.join(map(itemgetter(1), ctxs)))
                demo = f'{ref}\nQuestion: {q}\nAnswer (with Search): {a}'
            else:
                demo = f'Question: {q}\nAnswer (with Search): {a}'
            demos.append(demo)
        task = self.instruction['task']
        ret = self.instruction['retrieval'] + '\n\n' + '\n\n'.join(demos)
        ensemble = self.instruction['ensemble']
        return task, ret, ensemble
