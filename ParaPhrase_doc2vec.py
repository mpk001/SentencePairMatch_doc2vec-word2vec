
import numpy as np
import pandas as pd
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

class ParaPhrase_word2vec:

    def load_doc2vec(self, word2vecpath):
        f = open(word2vecpath)
        embeddings_index = {}
        count = 0
        for line in f:
            # count += 1
            # if count == 10000: break
            values = line.split('\t')
            id = values[0]
            print id
            coefs = np.asarray(values[1].split(), dtype='float32')
            embeddings_index[int(id)+1] = coefs
        f.close()
        print('Total %s word vectors.' % len(embeddings_index))

        return embeddings_index


    def load_data(self, datapath):
        data_train = pd.read_csv(datapath, sep='\t', encoding='utf-8')
        print data_train.shape

        qid1 = []
        qid2 = []
        labels = []
        count = 0
        for idx in range(data_train.id.shape[0]):
        # for idx in range(400):
        #     count += 1
        #     if count == 10: break
            print idx
            q1 = data_train.qid1[idx]
            q2 = data_train.qid2[idx]

            print q1
            qid1.append(q1)
            qid2.append(q2)
            labels.append(data_train.is_duplicate[idx])

        return qid1, qid2, labels

    def sentence_represention(self, qid, embeddings_index):
        vectors = np.zeros((len(qid), 100))
        for i in range(len(qid)):
            print i
            vectors[i] = embeddings_index.get(qid[i])

        return vectors

    def consin_distance(self, vectors1, vectors2):
        sim = []
        for i in range(vectors1.shape[0]):
            vec1 = vectors1[i]
            vec2 = vectors2[i]
            vec0 = np.dot(vec1, vec2)
            vec1 = np.power(vec1, 2)
            vec2 = np.power(vec2, 2)
            vec1 = np.sum(vec1)
            vec2 = np.sum(vec2)
            vec1 = np.sqrt(vec1)
            vec2 = np.sqrt(vec2)
            sim.append(vec0/(vec1 * vec2))

        return sim

    def write2file(self, texts1, texts2, sim, labels, outfile):
        f = file(outfile, "w+")
        for i in range(len(sim)):
            f.writelines(str(sim[i]) + '\t' + str(labels[i]) + '\n')

    # if __name__ == '__main__':
    #     texts1, texts2, labels = load_data(datapath)
    #     embeddings_index =load_word2vec(word2vecpath=word2vecpath)
    #     vectors1 = sentence_represention(texts1, embeddings_index)
    #     vectors2 = sentence_represention(texts2, embeddings_index)
    #     sim = consin_distance(vectors1, vectors2)
    #     write2file(texts1, texts2, sim, outfile=outfile)

if __name__ == '__main__':
    # doc2vecpath = "D:/dataset/quora/vector2/quora_duplicate_question_doc2vec_100.vector"
    # # word2vecpath = "D:/dataset/cn_word2vec_100.vector"
    # datapath = 'D:/dataset/quora/quora_duplicate_questions_Chinese_seg.tsv'
    # outfile = 'D:/dataset/quora/quora_duplicate_questions_Chinese_sim_doc2vec.tsv'
    # # outfile = 'D:/dataset/quora/quora_duplicate_questions_Chinese_sim_word2vec_1.tsv'

    doc2vecpath = "D:/dataset/quora/vector_english/quora_duplicate_question_doc2vec_100.vector"
    datapath = 'D:/dataset/quora/quora_duplicate_questions.tsv'
    outfile = 'D:/dataset/quora/quora_duplicate_questions_English_sim_doc2vec.tsv'
    para = ParaPhrase_word2vec()
    qid1, qid2, labels = para.load_data(datapath)
    embeddings_index = para.load_doc2vec(word2vecpath=doc2vecpath)
    vectors1 = para.sentence_represention(qid1, embeddings_index)
    vectors2 = para.sentence_represention(qid2, embeddings_index)
    sim = para.consin_distance(vectors1, vectors2)
    para.write2file(qid1, qid2, sim, labels, outfile=outfile)