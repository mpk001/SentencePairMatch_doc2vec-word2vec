#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec, doc2vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    # if len(sys.argv) < 4:
    #     print globals()['__doc__'] % locals()
    #     sys.exit(1)
    # inp, outp1, outp2 = sys.argv[1:4]

    # inp = "D:/data/corpus/en_wikifulltext/encorp_fulltext"
    # outp2 = "D:/data/corpus/en_wikifulltext/en_word2vec_100.vector"

    inp = "D:/dataset/cn.corpus"
    outp2 = "D:/dataset/cn_word2vec_100.vector"

    model = Word2Vec(LineSentence(inp), size=100, window=10, min_count=3,
            workers=multiprocessing.cpu_count(), sg=1, iter=10, negative=20)

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    # model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)