# Code taken and adapted from https://github.com/tejank10/Spam-or-Ham/blob/master/spam_ham.ipynb

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log
import nltk
nltk.download('punkt')

def create_word_cloud(full_df, is_troll_label):
    words = ' '.join(full_df[full_df.is_troll == is_troll_label].content.values)
    wc = WordCloud(width = 512,height = 512).generate(words)
    plt.figure(figsize = (10, 8), facecolor = 'k')
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.show()

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words

class TrollClassifier:
    def __init__(self, train_data, method='tf-idf'):
        self.mails, self.labels = train_data['content'].reset_index(drop=True), train_data['is_troll'].reset_index(drop=True)
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_troll = dict()
        self.prob_non_troll = dict()
        for word in self.tf_troll:
            self.prob_troll[word] = (self.tf_troll[word] + 1) / (self.troll_words + \
                                                                len(list(self.tf_troll.keys())))
        for word in self.tf_non_troll:
            self.prob_non_troll[word] = (self.tf_non_troll[word] + 1) / (self.non_troll_words + \
                                                                len(list(self.tf_non_troll.keys())))
        self.prob_troll_mail, self.prob_non_troll_mail = self.troll_mails / self.total_mails, self.non_troll_mails / self.total_mails 


    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]
        self.troll_mails, self.non_troll_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.troll_mails + self.non_troll_mails
        self.troll_words = 0
        self.non_troll_words = 0
        self.tf_troll = dict()
        self.tf_non_troll = dict()
        self.idf_troll = dict()
        self.idf_non_troll = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.mails.values[i])
            count = list() #To keep track of whether the word has ocured in the message or not.
                           #For IDF
            for word in message_processed:
                if self.labels.values[i]:
                    self.tf_troll[word] = self.tf_troll.get(word, 0) + 1
                    self.troll_words += 1
                else:
                    self.tf_non_troll[word] = self.tf_non_troll.get(word, 0) + 1
                    self.non_troll_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_troll[word] = self.idf_troll.get(word, 0) + 1
                else:
                    self.idf_non_troll[word] = self.idf_non_troll.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_troll = dict()
        self.prob_non_troll = dict()
        self.sum_tf_idf_troll = 0
        self.sum_tf_idf_non_troll = 0
        for word in self.tf_troll:
            self.prob_troll[word] = (self.tf_troll[word]) * log((self.troll_mails + self.non_troll_mails) \
                                                          / (self.idf_troll[word] + self.idf_non_troll.get(word, 0)))
            self.sum_tf_idf_troll += self.prob_troll[word]
        for word in self.tf_troll:
            self.prob_troll[word] = (self.prob_troll[word] + 1) / (self.sum_tf_idf_troll + len(list(self.prob_troll.keys())))
            
        for word in self.tf_non_troll:
            self.prob_non_troll[word] = (self.tf_non_troll[word]) * log((self.troll_mails + self.non_troll_mails) \
                                                          / (self.idf_troll.get(word, 0) + self.idf_non_troll[word]))
            self.sum_tf_idf_non_troll += self.prob_non_troll[word]
        for word in self.tf_non_troll:
            self.prob_non_troll[word] = (self.prob_non_troll[word] + 1) / (self.sum_tf_idf_non_troll + len(list(self.prob_non_troll.keys())))
            
    
        self.prob_troll_mail, self.prob_non_troll_mail = self.troll_mails / self.total_mails, self.non_troll_mails / self.total_mails 
                    
    def classify(self, processed_message):
        ptroll, pnon_troll = 0, 0
        for word in processed_message:                
            if word in self.prob_troll:
                ptroll += log(self.prob_troll[word])
            else:
                if self.method == 'tf-idf':
                    ptroll -= log(self.sum_tf_idf_troll + len(list(self.prob_troll.keys())))
                else:
                    ptroll -= log(self.troll_words + len(list(self.prob_troll.keys())))
            if word in self.prob_non_troll:
                pnon_troll += log(self.prob_non_troll[word])
            else:
                if self.method == 'tf-idf':
                    pnon_troll -= log(self.sum_tf_idf_non_troll + len(list(self.prob_non_troll.keys()))) 
                else:
                    pnon_troll -= log(self.non_troll_words + len(list(self.prob_non_troll.keys())))
            ptroll += log(self.prob_troll_mail)
            pnon_troll += log(self.prob_non_troll_mail)
        return ptroll >= pnon_troll
    
    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result

def metrics(labels, predictions):
    labels = labels.values
    predictions = predictions.values
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
#     precision = true_pos / (true_pos + false_pos)
#     recall = true_pos / (true_pos + false_neg)
#     Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

#     print("Precision: ", precision)
#     print("Recall: ", recall)
#     print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)