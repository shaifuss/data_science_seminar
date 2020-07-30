#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import Counter
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
import os


def get_scores(model_dict, vocab):
    for (method, output_dict) in model_dict.items():
        output_dict['scores'] = []
        topics = output_dict['topics']
        for topic in topics:
            score = 0
            for (i, word) in reversed(list(enumerate(reversed(topic)))):
                if word == 'pizza':  # most important word is worth double
                    score += (i + 1) * 2
                elif word in vocab:
                    score += i + 1
            output_dict['scores'].append(score)
        print 'method: {}, topic scores: {}'.format(method,
                model_dict[method]['scores'])
        print 'method: {}, coherence: {}'.format(method,
                model_dict[method]['coherence'])
        if 'BERT' in method:
            print 'method: {}, silhouette: {}'.format(method,
                    model_dict[method]['silhouette'])
    return model_dict


def visualize(model):
    """
    Visualize the result for the topic model by 2D embedding (UMAP)
    :param model: Topic_Model object
    """

    if model.method == 'LDA':
        return
    reducer = umap.UMAP()
    print 'Calculating UMAP projection ...'
    vec_umap = reducer.fit_transform(model.vec[model.method])
    plot_proj(vec_umap, model.cluster_model.labels_)


def get_topic_words_hybrid(token_lists, labels, k=None):
    """
    get top words within each topic from clustering results
    """

    if k is None:
        k = len(np.unique(labels))
    topics = ['' for _ in range(k)]
    for (i, c) in enumerate(token_lists):
        topics[labels[i]] += ' ' + ' '.join(c)
    word_counts = list(map(lambda x: Counter(x.split()).items(),
                       topics))

    # get sorted word counts

    word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1],
                       reverse=True), word_counts))

    # get topics

    topics = list(map(lambda x: list(map(lambda x: x[0], x[:5])),
                  word_counts))
    return topics


def get_topic_words_lda(model):
    return CoherenceModel.top_topics_as_word_lists(model=model.ldamodel,
            dictionary=model.dictionary, topn=5)


def get_coherence(model, token_lists, measure='c_v'):
    """
    Get model coherence from gensim.models.coherencemodel
    :param model: Topic_Model object
    :param token_lists: token lists of docs
    :param topics: topics as top words
    :param measure: coherence metrics
    :return: coherence score
    """

    if model.method == 'LDA':
        cm = CoherenceModel(model=model.ldamodel, texts=token_lists,
                            corpus=model.corpus,
                            dictionary=model.dictionary,
                            coherence=measure)
        print CoherenceModel.top_topics_as_word_lists(model=model.ldamodel,
                dictionary=model.dictionary, topn=5)
    else:
        topics = get_topic_words_hybrid(token_lists,
                model.cluster_model.labels_)
        print 'Topics are:\n{}'.format(topics)
        cm = CoherenceModel(topics=topics, texts=token_lists,
                            corpus=model.corpus,
                            dictionary=model.dictionary,
                            coherence=measure)
    return cm.get_coherence()


def get_silhouette(model):
    """
    Get silhouette score from model
    :param model: Topic_Model object
    :return: silhouette score
    """

    if model.method == 'LDA':
        return
    lbs = model.cluster_model.labels_
    vec = model.vec[model.method]
    return silhouette_score(vec, lbs)


def plot_proj(embedding, lbs):
    """
    Plot UMAP embeddings
    :param embedding: UMAP (or other) embeddings
    :param lbs: labels
    """

    n = len(embedding)
    counter = Counter(lbs)
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i],
                 '.', alpha=0.5, label='cluster {}: {:.2f}%'.format(i,
                 counter[i] / n * 100))
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='-', linewidth=0.25)
