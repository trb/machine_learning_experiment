import requests
import re
import os
import random
import numpy as np
import pylab as pl
import hashlib
import zlib
import json


from pyquery import PyQuery as pq
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn import ensemble

    
def cached_request(url):
    m = hashlib.md5()
    m.update(url)
    hash = m.hexdigest()
    
    try:
        with open(os.path.join('cache', hash)) as f:
            return f.read()
    except IOError:
        html = requests.get(url).content
        open(os.path.join('cache', hash), 'w').write(html)
        return html

    
def get_coordinates(address):
    url = 'http://maps.googleapis.com/maps/api/geocode/json?address={0}&sensor=false'
    url = url.format(address)
    
    response = cached_request(url)
    response = json.loads(response)
    
    location = response['results'][0]['geometry']['location']
    
    return (location['lat'], location['lng'])


class Post(object):
    def __init__(self, url):
        self.url = url
        self.content = cached_request(url)
        self.selector = pq(self.content)
        self.title = self.selector('title').text()
        self.body = self.selector('#userbody').text()
        
        price = self.selector('.postingtitle').text()
        if price is not None:
            price = re.match('\$([0-9]+)', price)
            if price is not None:
                price = price.groups()[0]
                
        size = re.search('(\d+?)ft', self.content)
        if size:
            size = size.groups()[0]
            
        location = re.search('q=loc%3A\+(.+?)">google\smap', self.content)
        if location is not None:
            location = location.groups()[0]
            
            try:
                location = get_coordinates(location)
            except:
                location = None
        
        self.price = price
        self.size = size
        self.location = location
        self.images = [ node.get('href') for node in self.selector('.tn a') ]
        
    def has_features(self):
        if not self.location:
            return False
        
        if not self.price:
            return False
        
        if not self.size:
            return False
        
        return True
        
    def get_features(self):
        location = self.location or (0, 0)
        return [ float(len(self.images)), float(self.price or 0), float(self.size or 0), float(location[0]), float(location[1]) ]
        #return [ float(self.price or 0) ]
        #return [ location[0], location[1] ]
        
    
def get_posts():
    likes = [ (Post(url), 1) for url in open('data/like').read().split('\n') ]
    dislikes = [ (Post(url), 0) for url in open('data/dislike').read().split('\n') ]
    
    return likes + dislikes

def get_posts_test():
    return [ Post(url) for url in open('data/test').read().split('\n') ]


def normalize_posts(posts):
    return np.array([
                     [post.get_features(), class_] for post, class_ in posts if post.has_features()
                     ])


def get_train_data(normalized_posts):
    data = normalized_posts[:, 0]
    count = int(len(data) / 2)
    return np.array(map(np.array, data[:count]))


def get_train_targets(normalized_posts):
    targets = normalized_posts[:, 1]
    count = int(len(targets) / 2)
    return np.array(map(np.array, targets[:count]))


def get_test_data(normalized_posts):
    data = normalized_posts[:, 0]
    count = int(len(data) / 2)
    return np.array(map(np.array, data[count:]))


def get_test_targets(normalized_posts):
    targets = normalized_posts[:, 1]
    count = int(len(targets) / 2)
    return np.array(map(np.array, targets[count:]))


def train(classifier, normalized_posts):
    data = get_train_data(normalized_posts)
    classifier.fit(data,
                   get_train_targets(normalized_posts))
    
    return classifier


def predict(classifier, normalized_posts):
    return classifier.predict(get_test_data(normalized_posts))


def evaluate_prediction(prediction, target):
    hits = 0
    for i, class_ in enumerate(prediction):
        if target[i] == class_:
            hits += 1
    
    return (float(hits)/len(prediction)) * 100


def evaluate_classifier(classifier, runs=10):
    posts = get_posts()
    total_score = 0.
    for run in range(runs):
        random.shuffle(posts)
        normalized_posts = normalize_posts(posts)
        classifier = train(classifier, normalized_posts)
        total_score += classifier.score(get_test_data(normalized_posts),
                                        get_test_targets(normalized_posts))

    return (total_score / runs) * 100


def test_against_normal_data(classifier, posts):
    features = np.array([ np.array(post.get_features()) for post in posts ])
    result = classifier.predict(features)
    for i, r in enumerate(result):
        if r == 1:
            print posts[i].url


def main():
    #classifier = KNeighborsClassifier() # 58%
    classifier = svm.LinearSVC(dual=False, class_weight={0: 2, 1: 1}) # 71%
    #classifier = BernoulliNB() # 46%
    #classifier = tree.DecisionTreeClassifier() #68%
    #classifier = SGDClassifier(n_iter=50) #59%
    #classifier = ensemble.RandomForestClassifier(n_estimators=80, max_features='auto') #71%
    #classifier = ensemble.ExtraTreesClassifier()  #76%
    posts = get_posts()
    print evaluate_classifier(classifier, 1000)
    #n_posts = normalize_posts(posts)
    #classifier.fit(np.array(map(np.array, n_posts[:, 0])), np.array(n_posts[:, 1]))
    #test_against_normal_data(classifier, get_posts_test())

if __name__ == '__main__':
    main()
