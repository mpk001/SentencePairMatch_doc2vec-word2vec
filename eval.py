#-*- coding: UTF-8 -*-

import os, sys


def get_prf(fpred, thre=0.1):
    """
    get Q-A level and Q level precision, recall, fmeasure
    """
    right_num = 0
    count = 0
    with open(fpred, "rb") as f:
        for line in f:
            count += 1
            print count,line
            parts = line.split('\r\n')[0].split('\t')
            p, l = float(parts[0]), float(parts[1])
            if p > thre and l == 1.0:#预测label为1
                right_num += 1
            elif p < thre and l == 0.0:
                right_num += 1

    return float(right_num) / count



if __name__ == "__main__":
    # refname, predname = sys.argv[1], sys.argv[2]
    thre = 0.8
    # if len(sys.argv) > 3:
    #     thre = float(sys.argv[3])
    # predname = 'D:/dataset/quora/quora_duplicate_questions_Chinese_sim_doc2vec.tsv'
    # predname = 'D:/dataset/quora/quora_duplicate_questions_Chinese_sim_word2vec.tsv'

    # predname = 'D:/dataset/quora/quora_duplicate_questions_English_sim_doc2vec.tsv'
    predname = 'D:/dataset/quora/quora_duplicate_questions_English_sim_word2vec.tsv'
    result = get_prf(predname, thre=thre)

    print "WikiQA Question Triggering: %.4f" %(result)
