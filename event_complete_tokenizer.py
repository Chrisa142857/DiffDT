import json

with open('icd10-complete_pure_vocab.json', 'r') as jsonf:
    tokenizer = json.load(jsonf)
    
def tokenizer_encode(txt):
    with open('icd10-complete_pure_vocab.json', 'r') as jsonf:
        tokenizer = json.load(jsonf)
    input_ids = []
    seq_posid = []
    posid = 0
    for items in txt.split(' '):
        for item in items.split('-'):
            if item not in tokenizer:
                i = len(tokenizer)
            else:
                i = tokenizer[item]
            input_ids.append(i)
            seq_posid.append(posid)
        posid += 1
    return input_ids, seq_posid
    
def tokenizer_decode(input_ids):
    with open('icd10-complete_pure_vocab.json', 'r') as jsonf:
        tokenizer = json.load(jsonf)
    tokenizer_re = {v: k for k, v in tokenizer.items()}
    txt = []
    for i in input_ids:
        if i not in tokenizer_re:
            code = '<unk>'
        else:
            code = tokenizer_re[i]
        txt.append(code)
    return txt

vocab_size = len(tokenizer)
