import os
import re
import collections
from tqdm import tqdm
from matplotlib import pyplot as plt



def get_vocab(data):
    vocab = collections.defaultdict(int)
    for word in data.strip().split():
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


if __name__ == '__main__':
    
    # training
    with open('./bpe-training-data.txt','r', encoding='utf-8') as f:
        data = f.read()
    data = data.replace('.',' ')
    data = data.replace(',',' ')
    data = data.replace('\n',' ')
    vocab = get_vocab(data)
    tokens = get_tokens(vocab)
    
    bar = tqdm(total=0)
    iters = 0
    record = []
    merge_rules = []
    while True:
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        if pairs[best] < 2:
            break
        new_token = ''.join(best)
        vocab = merge_vocab(best, vocab)
        tokens[new_token] = pairs[best]
        tokens[best[0]] -= pairs[best]
        tokens[best[1]] -= pairs[best]
        if tokens[best[0]] == 0:
            del tokens[best[0]]
        if tokens[best[1]] == 0:
            del tokens[best[1]]
        # report len of token
        bar.set_postfix({'len(tokens)': len(tokens), 'new_token': new_token.ljust(10), 'bset': pairs[best]})
        bar.update(1)
        iters += 1
        record.append((iters, len(tokens)))
        merge_rules.append(best)
        
    # draw
    x = [r[0] for r in record]
    y = [r[1] for r in record]
    plt.plot(x, y)
    plt.xlabel('iteration')
    plt.ylabel('len(tokens)')
    plt.savefig('bpe.png')
    plt.show()
    
    # report
    print("len(tokens):", len(tokens))
    len_data = 0
    for i in vocab:
        len_data += len(i.split())
    print("len(data):", len_data)
    
    # predict
    with open('./bpe-testing-article.txt','r', encoding='utf-8') as f:
        test_data = f.read()
    test_data = test_data.split('\n')
    for sentence in test_data:
        if sentence == '' or sentence == ' ' or sentence == '\n':
            continue
        # preprocess sentence
        temp_vocab = get_vocab(sentence)
        for rule in merge_rules:
            temp_vocab = merge_vocab(rule, temp_vocab)
        # rebuild sentence
        sentence = ''
        for word in temp_vocab:
            sentence += word.replace('</w>','') + ' '
        print(sentence)
        print(len(sentence.split()))
            
    
    