import os
import sys
import json
from collections import OrderedDict

import numpy as np
from keras.models import Model

from matchzoo.inputs.point_generator import PointGenerator
from matchzoo.utils import *


class Inference:

    def __init__(self, config, model_prefix='/home/hs/gits/MatchZoo'):
        self.config = config
        self.model_prefix = model_prefix

        self.input_predict_conf = None
        self.model = None
        self.word_dict = None

    @staticmethod
    def _load_embedding(share_input_conf):
        _share_input_conf = share_input_conf.copy()
        # collect embedding
        if 'embed_path' in _share_input_conf:
            embed_dict = read_embedding(filename=_share_input_conf['embed_path'])
            _PAD_ = _share_input_conf['vocab_size'] - 1
            embed_dict[_PAD_] = np.zeros((_share_input_conf['embed_size'],), dtype=np.float32)
            embed = np.float32(
                np.random.uniform(-0.02, 0.02, [_share_input_conf['vocab_size'], _share_input_conf['embed_size']]))
            _share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed=embed)
        else:
            embed = np.float32(
                np.random.uniform(-0.2, 0.2, [_share_input_conf['vocab_size'], _share_input_conf['embed_size']]))
            _share_input_conf['embed'] = embed
        print('[Embedding] Embedding Load Done.', end='\n')
        return _share_input_conf

    @staticmethod
    def _process_input_tags(share_input_conf, input_conf):
        # list all input tags and construct tags config
        input_predict_conf = OrderedDict()
        for tag in input_conf.keys():
            if 'phase' not in input_conf[tag]:
                continue
            if input_conf[tag]['phase'] == 'PREDICT':
                input_predict_conf[tag] = {}
                input_predict_conf[tag].update(share_input_conf)
                input_predict_conf[tag].update(input_conf[tag])
        print('[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys()), end='\n')
        return input_predict_conf

    def _load_keras_model(self, config):
        global_conf = config["global"]
        model_type = global_conf['model_type']
        mo = None
        if model_type == 'JSON':
            mo = Model.from_config(config['model'])
        elif model_type == 'PY':
            model_config = config['model']['setting']
            model_config.update(config['inputs']['share'])
            sys.path.insert(0, config['model']['model_path'])

            model = import_object(config['model']['model_py'], model_config)
            mo = model.build()

        # Load Model
        weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])
        if not os.path.isfile(weights_file):

            if weights_file.split('/')[0] == '.':
                weights_file = weights_file[2:]
            weights_file = '{}/{}'.format(self.model_prefix, weights_file)

        mo.load_weights(weights_file)
        return mo

    def load_pretrained_model(self):
        # print(json.dumps(self.config, indent=2), end='\n')
        input_conf = self.config['inputs']
        share_input_conf = input_conf['share']

        share_input_conf = self._load_embedding(share_input_conf)
        input_conf['share'] = share_input_conf
        input_predict_conf = self._process_input_tags(share_input_conf, input_conf)
        self.input_predict_conf = input_predict_conf

        # ######## Read output config ########
        # output_conf = self.config['outputs']

        model = self._load_keras_model(self.config)
        return model

    @staticmethod
    def _make_generator_ingredient(str_pairs_list, word_dict):
        """

        Dataset: {tid: indexed string}
        rel_file: [int(label), tid1, tid2]
        :param str_pairs_list:
        :return:
        """

        def _indexing_sent(sent):
            indexed_sent = []
            for word in sent.split():
                if word not in word_dict:
                    indexed_sent.append(word_dict['UNK'])
                else:
                    indexed_sent.append(word_dict[word])
            return indexed_sent

        dataset = {}
        mapper = {}
        rel = []
        for pair in str_pairs_list:
            sent1, sent2 = pair
            if sent1 not in mapper:
                tid1 = 'T{}'.format(len(dataset))
                dataset[tid1] = _indexing_sent(sent1)
                mapper[sent1] = tid1
            else:
                tid1 = mapper[sent1]

            if sent2 not in mapper:
                tid2 = 'T{}'.format(len(dataset))
                dataset[tid2] = _indexing_sent(sent2)
                mapper[sent2] = True
            else:
                tid2 = mapper[sent2]

            rel.append([1, tid1, tid2])
        return dataset, rel, mapper

    def create_generator(self, str_pairs_list):
        if self.word_dict is None:
            self.word_dict = self._load_word_dict()
        dataset, rel, mapper = self._make_generator_ingredient(str_pairs_list, self.word_dict)
        config = self.input_predict_conf.copy()['predict']
        point_generator = PointGenerator(config, {'data': dataset, 'rel': rel})
        return point_generator

    def _load_word_dict(self):
        word_dict_dir = '{}/data/toy_example/classification/word_dict.txt'.format(self.model_prefix)
        word_dict = {}
        with open(word_dict_dir, 'r') as f:
            for line in f:
                word, idx = line.strip().split()
                word_dict[word] = int(idx)
        return word_dict

    def infer(self, str_pairs_list):
        if self.model is None:
            self.model = self.load_pretrained_model()
        generator = self.create_generator(str_pairs_list)

        y_preds = []
        for input_data, y_true in generator.get_batch_generator():
            y_pred = self.model.predict(input_data, batch_size=len(y_true))
            y_preds.append(y_pred)
            # y_preds.extend(list(y_pred[:, 1]))
        return y_preds
#
# if __name__ == '__main__':
#     with open('/home/hs/gits/MatchZoo/examples/toy_example/config/matchpyramid_classify.config', 'r') as f:
#         model_config = json.load(f)
#
#     inferrer = Inference(model_config)
#     inferrer.load_pretrained_model()
#     result = inferrer.infer([('뭐 하는 거야 ?', '뭐 하는 거니 ?'),
#                              ('뭐 하는 거야', '뭐 하는 거니'),
#                              ('어디 살아 ?', '어디 살아 요 ?')
#                              ])
#     print(result)
