from gensim import corpora, models, similarities
from pprint import pprint
import os
import os.path

if __name__ == '__main__':
    fileroot = os.path.join(os.getcwd(),'MeachinLearning','LDA','22.LDA_test.txt')
    f = open(fileroot)
    stop_list = set('for a of the end to in'.split())
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    print('Text=')
    pprint(texts)
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print('dictionary.token2id:')
    pprint(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('corpus=')
    pprint(corpus)
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print('TF-IDF:')
    for c in corpus_tfidf:
        print(c)
    print('\nLSI Model')
    lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary)
    topic_result = [a for a in lsi[corpus_tfidf]]
    print('\nLSI Topics:')
    pprint(topic_result)
    pprint(lsi.print_topics(num_topics=2, num_words=5))
    Similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])
    print('Similarity:')
    pprint(list(Similarity))

    print('\n LDA Model:')
    num_topics = 2
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, alpha='auto', eta='auto', minimum_probability=0.001)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print('Document-Topic:\n')
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print(doc_topic)
    for topic_id in range(num_topics):
        print('Topic:{}'.format(topic_id))
        pprint(lda.show_topic(topic_id))
    Similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print('Similarity:')
    pprint(list(Similarity))

    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    print('\n\nUSE WITH CARE--\nHDA Model:')
    pprint(topic_result)
    print('HDA Topics:')
    print(hda.print_topics(num_topics=2, num_words=5))


    