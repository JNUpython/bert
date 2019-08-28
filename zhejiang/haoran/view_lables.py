import os
import codecs
import pickle
if os.path.exists("/Users/mo/Documents/github_projects/zhijiang/JNU/bert/output/label2id.pkl"):
    with codecs.open("/Users/mo/Documents/github_projects/zhijiang/JNU/bert/output/label2id.pkl", 'rb') as rf:
        labels = pickle.load(rf)
        print(labels)