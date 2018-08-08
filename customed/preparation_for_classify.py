import sys
import argparse
from tqdm import tqdm
sys.path.append('../matchzoo/inputs')
sys.path.append('../matchzoo/utils')
sys.path.append('/home/hs/mudbox/ml')

from pingpong.utils import get_ingredient_factory
from preparation import *


class Preprocess:

    def __init__(self, corpus, base_dir, word_dict_dir, min_count):
        self.corpus = corpus
        self.base_dir = base_dir
        self.word_dict_dir = word_dict_dir
        self.min_count = min_count

        self.word_dict = self.load_word_dict()

    def load_word_dict(self):
        word_dict = {'UNK': {'idx': 0, 'count': 10000000}}
        with open(self.word_dict_dir, 'r') as f:
            for idx, line in enumerate(f):
                word, count = line.strip().split('\t')
                count = int(count)
                if count > self.min_count or word == 'UNK':
                    word_dict[word] = {'idx': idx, 'count': count}
        return word_dict

    def word_indexing(self, tokenizer=False):
        indexed_corpus = {}
        for doc_id, sent in tqdm(self.corpus.items()):
            indexed_sent = []

            if tokenizer:
                sent = tokenizer.run(sent)

            for word in sent.split():
                try:
                    indexed_sent.append(str(self.word_dict[word]['idx']))
                except KeyError:
                    indexed_sent.append(str(self.word_dict['UNK']['idx']))
            indexed_corpus[doc_id] = ' '.join(indexed_sent)
        return indexed_corpus

    def save_word_dict(self):
        word_dict_path = self.base_dir + 'word_dict.txt'
        words_stats_path = self.base_dir + 'word_stats.txt'

        f_word_dict = open(word_dict_path, 'w')
        f_words_stats = open(words_stats_path, 'w')

        for word, word_info in self.word_dict.items():
            word_idx = word_info['idx']
            word_stat = word_info['count']
            f_word_dict.write('%s %s\n' % (word, str(word_idx)))
            f_words_stats.write('%s %s\n' % (word, str(word_stat)))

        f_word_dict.close()
        f_words_stats.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--word_dict', default='')
    parser.add_argument('--min_count', default=20)
    parser.add_argument('--tokenize', default=False)

    # args = parser.parse_args()
    # base_dir = args.base_dir
    # word_dict_dir = args.word_dict
    # min_count = args.min_count
    # tokenize = args.tokenize

    base_dir = '/home/hs/mudbox/query_expansion/matchzoo/'
    word_dict_dir = base_dir + 'vocab.txt'
    min_count = 20
    tokenize = True

    if base_dir is None or word_dict_dir is None:
        print('DIR@!!!')
        return

    if tokenize:
        factory = get_ingredient_factory()
        tokenizer = factory.get_tokenized_pipeline()

    prepare = Preparation()
    train_file = base_dir + 'train.txt'
    valid_file = base_dir + 'val.txt'
    test_file = base_dir + 'test.txt'
    corpus, rels_train, rels_valid, rels_test = prepare.run_with_train_valid_test_corpus(train_file,
                                                                                         valid_file,
                                                                                         test_file)
    print('total corpus : %d ...' % (len(corpus)))
    print('train relations : %d ...' % (len(rels_train)))
    print('valid relations : %d ...' % (len(rels_valid)))
    print('test relations : %d ...' % (len(rels_test)))

    prepare.save_corpus(base_dir + 'corpus.txt', corpus)  # corpus: {tid: sent}
    prepare.save_relation(base_dir + 'relation_train.txt', rels_train)
    prepare.save_relation(base_dir + 'relation_valid.txt', rels_valid)
    prepare.save_relation(base_dir + 'relation_test.txt', rels_test)

    preprocess = Preprocess(corpus, base_dir, word_dict_dir, min_count)
    indexed_corpus = preprocess.word_indexing(tokenizer=tokenizer)

    f_out = open(base_dir + 'corpus_preprocessed.txt', 'w')
    for doc_id, indexed_sent in indexed_corpus.items():
        f_out.write('%s %d %s\n' % (doc_id, len(indexed_sent), indexed_sent))
    f_out.close()

    preprocess.save_word_dict()
    print('preprocess finished ...')


if __name__ == '__main__':
    main()
