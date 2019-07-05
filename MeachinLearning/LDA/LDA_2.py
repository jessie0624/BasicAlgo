import numpy as np  
from gensim import corpora, models, similarities
from pprint import pprint
import time
import os
import os.path

def load_stopword():
    fileroot = os.path.join(os.getcwd(),'MeachinLearning','LDA','22.stopword.txt')
    f_stop = open(fileroot)
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

if __name__ == '__main__':
    print(u'初始化停止词列表 --')
    t_start = time.time()
    stop_words = load_stopword()

    print(u'开始读入语料数据 --')
    data = os.path.join(os.getcwd(), 'MeachinLearning', 'LDA', '22.news.dat')
    f = open(data,encoding='utf-8')
    texts = [[word for word in line.strip().lower().split() if word not in stop_words] for line in f]
    print('读入语料数据完成, 用时%.3f秒'%(time.time() - t_start))
    f.close()
    M = len(texts)
    print('文本数目：%d个'% M)

    print('正在建立词典 --')
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print('正在计算文本向量 --')
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('正在计算本档TF-IDF --')
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print('建立文档TF-IDF完成，用时%.3f秒'%(time.time() - t_start))
    print('LDA 模型拟合推断--')
    num_topics = 30 ###手动设定的参数
    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,alpha=0.001, eta=0.01, minimum_phi_value=0.001,update_every=1,chunksize=100, passes=1)
    print('LDA模型完成，训练时间为\t%.3f秒' %(time.time() - t_start))
    ##随机打印某10个文档主题
    num_show_topic = 10
    print('10个文档的主题分布：')
    doc_topics = lda.get_document_topics(corpus_tfidf)
    idx = np.arange(M)
    np.random.shuffle(idx)
    idx = idx[:10]
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:,1])
        topic_idx = topic_distribute.argsort()[:-num_show_topic-1:-1]
        print('第%d个文档的前%d个主题'%(i, num_show_topic))
        print(topic_distribute[topic_idx])
    num_show_term = 7
    print('每个主题的词分布：')
    for topic_id in range(num_topics):
        print('主题%d:\t'% topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:,0].astype(np.int)
        print('词:\t')
        for t in term_id:
            print(dictionary.id2token[t])
        
