"""Frozen ConFiQA prompt and scorer, extracted without semantic changes."""

import json
import re
import string
from collections import Counter

def norm(value):
    if value is None:
        return ''
    if not isinstance(value, str):
        value = str(value)
    value = value.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value).split())

def score(prediction, answers):
    answers = [a for a in answers if a is not None and str(a).strip()]
    if not answers:
        return {'EM': 0.0, 'F1': 0.0}
    em = max(float(norm(prediction)==norm(a)) for a in answers)
    f1 = 0.
    for answer in answers:
        p,a = norm(prediction).split(),norm(answer).split()
        overlap = sum((Counter(p)&Counter(a)).values())
        value = 2*overlap/(len(p)+len(a)) if p or a else 1.
        f1 = max(f1,value)
    return {'EM':em,'F1':f1}

def rows(path):
    return [json.loads(x) for x in path.open() if x.strip()]

def leading_answer(text):
    """Frozen label-blind first-segment parser. Raw score is always also reported."""
    value=re.split(r'\n|question\s*:|task\s*:',text.strip(),maxsplit=1,flags=re.I)[0]
    value=re.sub(r'^answer\s*:\s*','',value,flags=re.I).strip().rstrip('.').strip()
    for left,right in [('**','**'),('"','"'),('`','`')]:
        if value.startswith(left) and value.endswith(right) and len(value)>2*len(left):
            value=value[len(left):-len(right)].strip()
    return value

def includes(text,aliases):
    return any(norm(a) and (' '+norm(a)+' ') in (' '+norm(text)+' ') for a in aliases)

def outcome(text,row,intent):
    answer=leading_answer(text)
    pa=row['prior_aliases']; ca=row['context_aliases']
    p=bool(score(answer,pa)['EM']); c=bool(score(answer,ca)['EM'])
    both=includes(answer,pa) and includes(answer,ca)
    category='both' if both else ('prior' if p else 'context' if c else 'other')
    return {'raw_response':text,'answer':answer,
            'parsed':score(answer,pa if intent=='prior' else ca),
            'raw':score(text,pa if intent=='prior' else ca),'source_category':category}

RULES = {
    'prior': 'Ignore the passage when choosing your answer. Answer the question from your own stored knowledge, even if the passage disagrees. Return only the short answer.',
    'context': 'Answer the question using only the passage, even if it disagrees with your own knowledge. Return only the short answer.',
}

DEMOS = (
    ('Water freezes at 80 degrees Celsius.', 'At what temperature does water freeze in degrees Celsius?', '0', '80'),
    ('A triangle has seven sides.', 'How many sides does a triangle have?', 'three', 'seven'),
)

def prompt(question, context, intent, variant):
    def block(q, c, i):
        return f'Passage:\n{c}\nQuestion: {q}\nInstruction: {RULES[i]}\nAnswer:'
    prefix = ''
    if variant == 'fewshot':
        prefix = '\n\n'.join(block(q, c, i) + ' ' + (p if i == 'prior' else a)
                               for c, q, p, a in DEMOS for i in ('prior', 'context')) + '\n\n'
    elif variant != 'tail':
        raise ValueError(variant)
    return prefix + block(question, context, intent)
