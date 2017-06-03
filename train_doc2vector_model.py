#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing

from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedLineDocument

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

    # inp = "D:/dataset/quora/quora_duplicate_questionslilst_Chinese.txt"
    # outp1 = "D:/dataset/quora/vector2/quora_duplicate_question_doc2vec_model.txt"
    # outp2 = "D:/dataset/quora/vector2/quora_duplicate_question_word2vec_100.vector"
    # outp3 = "D:/dataset/quora/vector2/quora_duplicate_question_doc2vec_100.vector"
    inp = "D:/dataset/quora/quora_duplicate_questionslilst_English.tsv"
    outp1 = "D:/dataset/quora/vector_english/quora_duplicate_question_doc2vec_model.txt"
    outp2 = "D:/dataset/quora/vector_english/quora_duplicate_question_word2vec_100.vector"
    outp3 = "D:/dataset/quora/vector_english/quora_duplicate_question_doc2vec_100.vector"

    model = Doc2Vec(TaggedLineDocument(inp), size=100, window=5, min_count=0, workers=multiprocessing.cpu_count(), dm=0,
                    hs=0, negative=10, dbow_words=1, iter=10)

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)#save dov2vec model
    model.wv.save_word2vec_format(outp2, binary=False)#save word2vec向量
    #保存doc2vector向量
    outid = file(outp3, 'w')
    print "doc2vecs length:", len(model.docvecs)
    for id in range(len(model.docvecs)):
        outid.write(str(id)+"\t")
        for idx,lv in enumerate(model.docvecs[id]):
            outid.write(str(lv)+" ")
        outid.write("\n")

    outid.close()

