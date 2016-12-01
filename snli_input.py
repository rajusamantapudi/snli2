import sys
import os as os
import numpy as np
import json
# can be sentence or word
input_mask_mode = "word"

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def init_snli(fname):
    
    print "==> Loading test from %s" % fname
    data_all = []
    k = 0
    with open(fname, 'rb') as f:
        for line in f:
            k+= 1
            line = line.decode('utf-8')
            line = line.lower()
            data = json.loads(line)
            if data['gold_label'] == '-':
            # ignore items without a gold label
                continue

            sentence1_parse = data['sentence1']
            sentence2_parse = data['sentence2']
            label = data['gold_label']

            t = {'sent1': sentence1_parse, 'sent2':sentence2_parse , 'label':label}
            data_all.append(t)

    print("Loaded {} data".format(len(data_all)))
    
    return data_all


def get_snli_raw():
    snli_train_raw = init_snli('data/snli_1.0/snli_1.0_train.jsonl')
    snli_dev_raw = init_snli('data/snli_1.0/snli_1.0_dev.jsonl')
    snli_test_raw = init_snli('data/snli_1.0/snli_1.0_test.jsonl')
    return snli_train_raw, snli_dev_raw, snli_test_raw

            
def load_glove(dim):
    word2vec = {}
    
    print "==> loading glove"
    with open(("./data/glove/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print "==> glove is loaded"
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def process_input(data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False):
    questions = []
    inputs = []
    answers = []
    input_masks = []

    label_map = {'entailment':1, 'neutral':0, 'contradiction':2}
    for x in data_raw:
        if split_sentences:
            inp = x["sent1"].lower().split(' . ') 
            inp = [w for w in inp if len(w) > 0]
            inp = [i.split() for i in inp]
        else:
            inp = x["sent1"].lower().split(' ') 
            inp = [w for w in inp if len(w) > 0]

        q = x["sent2"].lower().split(' ')
        q = [w for w in q if len(w) > 0]

        if split_sentences: 
            inp_vector = [[process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in s] for s in inp]
        else:
            inp_vector = [process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in inp]
                                    
        q_vector = [process_word(word = w, 
                                    word2vec = word2vec, 
                                    vocab = vocab, 
                                    ivocab = ivocab, 
                                    word_vector_size = embed_size, 
                                    to_return = "index") for w in q]
        
        if split_sentences:
            inputs.append(inp_vector)
        else:
            inputs.append(np.vstack(inp_vector).astype(floatX))

        questions.append(np.vstack(q_vector).astype(floatX))
        
        answers.append(label_map[x['label']])

        if not split_sentences:
            if input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
            elif input_mask_mode == 'sentence': 
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
            else:
                raise Exception("invalid input_mask_mode")
    
    return inputs, questions, answers, input_masks

def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)
    

def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(len(padded_sentences)-max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]),(0,0)), 'constant', constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding

def load_snli(config, split_sentences=False):
    vocab = {}
    ivocab = {}

    snli_train_raw, snli_val_raw, snli_test_raw = get_snli_raw()

    if config.word2vec_init:
        assert config.embed_size == 100
        word2vec = load_glove(config.embed_size)
    else:
        word2vec = {}

    # set word at index zero to be end of sentence token so padding with zeros is consistent
    process_word(word = "<eos>", 
                word2vec = word2vec, 
                vocab = vocab, 
                ivocab = ivocab, 
                word_vector_size = config.embed_size, 
                to_return = "index")

    print '==> get train inputs'
    train_data = process_input(snli_train_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)
    print '==> get validation inputs'
    val_data = process_input(snli_val_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)   
    print '==> get test inputs'
    test_data = process_input(snli_test_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)

    if config.word2vec_init:
        assert config.embed_size == 100
        word_embedding = create_embedding(word2vec, ivocab, config.embed_size)
    else:
        word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init, (len(ivocab), config.embed_size))

    inputs, questions, answers, input_masks = train_data if config.train_mode else test_data


    if split_sentences:
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)
        mask_lens = get_lens(input_masks)
        max_mask_len = np.max(mask_lens)

    q_lens = get_lens(questions)
    max_q_len = np.max(q_lens)
    max_input_len = min( np.max(input_lens), config.max_allowed_inputs)

    if config.train_mode:
        val_q_len = get_lens(val_data[1])
        val_input_len = get_lens(val_data[0])
        val_mask_len = get_lens(val_data[3])

        max_q_len = max(max_q_len, np.max(val_q_len))
        max_input_len = min( max(np.max(input_lens),np.max(val_input_len)), config.max_allowed_inputs)
        max_mask_len = max(max_mask_len, np.max(val_mask_len))

    #pad out arrays to max
    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        # here
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    questions = pad_inputs(questions, q_lens, max_q_len)

    answers = np.stack(answers)

    if config.train_mode:
        train = questions, inputs, q_lens, input_lens, input_masks, answers
        val_input_mask = pad_inputs(val_data[3], val_mask_len, max_mask_len, "mask")
        valid = pad_inputs(val_data[1], val_q_len, max_q_len), pad_inputs(val_data[0], val_input_len , max_input_len), val_q_len, val_input_len ,val_input_mask , np.stack(val_data[2])

        return train, valid, word_embedding, max_q_len, max_input_len, len(vocab)

    else:
        test = questions, inputs, q_lens, input_lens, input_masks, answers

        return test, word_embedding, max_q_len, max_input_len, len(vocab)


    
