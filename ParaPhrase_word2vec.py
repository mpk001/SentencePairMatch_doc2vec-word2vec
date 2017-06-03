
import numpy as np
import pandas as pd
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

class ParaPhrase_word2vec:

    def load_word2vec(self, word2vecpath):
        f = open(word2vecpath)
        embeddings_index = {}
        count = 0
        for line in f:
            # count += 1
            # if count == 10000: break
            values = line.split()
            word = values[0]
            print word
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Total %s word vectors.' % len(embeddings_index))

        return embeddings_index


    def load_data(self, datapath):
        data_train = pd.read_csv(datapath, sep='\t', encoding='utf-8')
        print data_train.shape

        texts1 = []
        texts2 = []
        labels = []
        count = 0
        for idx in range(data_train.id.shape[0]):
        # for idx in range(400):
        #     count += 1
        #     if count == 10: break
            print idx
            text1 = data_train.question1[idx]
            text2 = data_train.question2[idx]
            text1 = text1.encode('UTF-8')
            text2 = text2.encode('UTF-8')

            print text1
            texts1.append(text1)
            texts2.append(text2)
            labels.append(data_train.is_duplicate[idx])

        return texts1, texts2, labels

    def sentence_represention(self, texts, embeddings_index):
        vectors = np.zeros((len(texts), 100))
        for i in range(len(texts)):
            vec = [0.0 for x in range(100)]
            vec = np.array(vec)
            words = texts[i].split()
            for j in range(len(words)):
                print words[j]
                vector = embeddings_index.get(words[j])
                if vector is not None:
                    vec += vector
            vec /= len(words)
            vectors[i] = vec

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
    # # word2vecpath = "D:/dataset/quora/vector2/quora_duplicate_question_word2vec_100.vector"
    # word2vecpath = "D:/dataset/cn_word2vec_100.vector"
    # datapath = 'D:/dataset/quora/quora_duplicate_questions_Chinese_seg.tsv'
    # # outfile = 'D:/dataset/quora/quora_duplicate_questions_Chinese_sim_word2vec.tsv'
    # outfile = 'D:/dataset/quora/quora_duplicate_questions_Chinese_sim_word2vec_1.tsv'

    word2vecpath = "D:/dataset/quora/vector_english/quora_duplicate_question_word2vec_100.vector"
    # word2vecpath = "D:/dataset/cn_word2vec_100.vector"
    datapath = 'D:/dataset/quora/quora_duplicate_questions.tsv'
    outfile = 'D:/dataset/quora/quora_duplicate_questions_English_sim_word2vec.tsv'
    # outfile = 'D:/dataset/quora/quora_duplicate_questions_Chinese_sim_word2vec_1.tsv'
    para = ParaPhrase_word2vec()
    texts1, texts2, labels = para.load_data(datapath)
    embeddings_index = para.load_word2vec(word2vecpath=word2vecpath)
    vectors1 = para.sentence_represention(texts1, embeddings_index)
    vectors2 = para.sentence_represention(texts2, embeddings_index)
    sim = para.consin_distance(vectors1, vectors2)
    para.write2file(texts1, texts2, sim, labels, outfile=outfile)