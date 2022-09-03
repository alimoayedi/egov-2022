from abc import ABC, abstractmethod
import numpy as np
from small_text.query_strategies.exceptions import EmptyPoolException, PoolExhaustedException
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier


def disagreement(votes):
    return len(set(votes)) - 1


class QueryStrategy(ABC):
    """
    Code from https://github.com/webis-de/small-text
    """

    @abstractmethod
    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        pass

    @staticmethod
    def _validate_query_input(indices_unlabeled, n):
        if len(indices_unlabeled) == 0:
            raise EmptyPoolException('No unlabeled indices available. Cannot query an empty pool.')

        if n > len(indices_unlabeled):
            raise PoolExhaustedException('Pool exhausted: {} available / {} requested'
                                         .format(len(indices_unlabeled), n))


class QBC(QueryStrategy):
    """
    implement Query by Committee strategy
    """

    def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)
        multi_label = _dataset.is_multi_label

        # train and test split
        train = _dataset[indices_labeled]
        test = _dataset[indices_unlabeled]

        if multi_label:
            y_train_t = train.y.toarray()
        else:
            y_train_t = train.y
        y_train_t = y_train_t.tolist()

        # init the committee
        member1 = clf
        if multi_label:
            member2 = MultiOutputClassifier(BernoulliNB(), n_jobs=-1)
            member3 = OneVsRestClassifier(SVC(C=1, gamma=0.1, kernel='linear', class_weight='balanced'), n_jobs=-1)
        else:
            member2 = BernoulliNB()
            member3 = SVC(C=1, gamma=0.1, kernel='linear', class_weight='balanced')

        # train member 2 and 3
        member2 = member2.fit(train.x, y_train_t)
        member3 = member3.fit(train.x, y_train_t)

        # predict votes
        if multi_label:
            votes1 = member1.predict(test).toarray()
        else:
            votes1 = member1.predict(test)
        votes2 = member2.predict(test.x)
        votes3 = member3.predict(test.x)

        if multi_label:
            dis_labelwise = []
            for label in range(len(_dataset.target_labels)):
                v1_slice = votes1[:, label].reshape(-1)
                v2_slice = votes2[:, label].reshape(-1)
                v3_slice = votes3[:, label].reshape(-1)
                voting = np.vstack((v1_slice, v2_slice, v3_slice))
                disagreements = [disagreement(inst) for inst in voting.T]
                dis_labelwise.append(disagreements)
            m_dis = np.array(dis_labelwise)
            disagreements = m_dis.sum(axis=0)
        else:
            voting = np.vstack((votes1, votes2, votes3))
            disagreements = [disagreement(inst) for inst in voting.T]

        df = pd.DataFrame({'indices': indices_unlabeled, 'disagreement': disagreements})
        df = df.sort_values(by=['disagreement'], ascending=False)
        selection = df['indices'].to_numpy()[:n]

        return selection
