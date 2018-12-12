---
title: Detecting Trolls, Saving Democracy
---

![seas-iacs](pics/SEAS_IACS.png)

Welcome to our CS209a Project: Twitter Troll Detection!

## Group Members
- Joe Davidson
- Julien Laasri
- Abhimanyu Talwar
- Dylan Randle

## Project Statement & Objectives

### Background

We live in times of unprecedented technological advancement. Some of the greatest developments in
society and culture today are social media websites such as Facebook and Twitter. These vast online
networks are used ubiquitously around the globe for connecting with friends and sharing information.
While this undoubtedly provides a lot of positive utility in most cases, there are situations where
these unregulated social networks can be harmful. Most notably, Twitter was recently used by "troll"
agents with the aim of swaying public opinion and influencing the results of the 2016 U.S. election.

Thanks to the FiveThirtyEight organization, we have obtained a [dataset](https://github.com/fivethirtyeight/russian-troll-tweets) containing high-fidelity tweets
produced by trolls. This dataset contains nearly 3 million tweets sent from Twitter handles connected
to the Internet Research Agency (IRA), a Russian "troll factory" and a defendant in an indictment filed
by the Justice Department in February 2018, as part of special counsel Robert Mueller's Russia
investigation. The vast majority of the tweets in this dataset were posted from 2015 through 2017
(straddling the 2016 presidential election).

### Method

Using the FiveThirtyEight data as positive-troll examples, we have collected an equally large set of
tweets through various search queries relating to the 2016 election ([here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FPDI7IN)). We use these data as
negative-troll examples. We are well aware that this non­-troll dataset may in fact include trolls,
but these concerns are dampened by the fact that the overall population of trolls is relatively small
compared to regular users, and that Twitter actively removes tweets from troll users (resulting in an
even smaller proportion of tweets which come from trolls).

Following the results of our [EDA](https://joeddav.github.io/troll_classification/more_eda.html), we clean, temporally slice, and stratify our data such that the distribution of tweet post dates are approximately equal, as shown in [Cleaning](https://joeddav.github.io/troll_classification/Dataset%20Consolidation.html). When all is said and done, the temporal distribution of the data looks as follows.

![temporal-histogram](pics/temporal_histogram.png)

### Models

The general goal in this project is to predict $P(troll|tweet)=P(y|X)$, with classification accuracy as
our measure of performance. We test three methods for representing the text component of $X$:

- Bag-of-Words (BoW)

- Term-Frequency Inverse-Document-Frequency (TF-IDF)

- Pre-trained [Semantic Sentence Embeddings](https://joeddav.github.io/troll_classification/sentence_embedding_eda.html) (from [InferSent](https://github.com/facebookresearch/InferSent))

And we fit four different models:

- Baseline [Naive Bayes](https://joeddav.github.io/troll_classification/naive_bayes.html)

- [Logistic Regression](https://joeddav.github.io/troll_classification/logistic_regression.html)

- [Support Vector Machine](https://joeddav.github.io/troll_classification/svm.html)

- Fully Connected [Neural Network](https://joeddav.github.io/troll_classification/neural_network.html)

### Extensions

To make things more interesting, we extend our problem in a few ways:

1. Train on tweets before time $T$, and test on tweets after time $T$
  - By doing this, we simulate the real-life scenario where we have collected training data and wish to
    predict troll/non-troll from new data, where the distribution of topics/content may have changed over time.

2. Exclude a proportion $X$% of labels (positive and negative), and try to re-label all those excluded labels
  - In this semi-supervised approach, we again simulate the real-life scenario where we do not have all of the
    data labeled, and instead have a subset, but wish to classify everything to be able to take appropriate
    action.

### Results

| Model | Standard Split | Temporal Split | Semi-Supervised |
|:-----:|:--------------:|:--------------:|:---------------:|
|Bag of Words | 0.0 | 0.0 | 0.0 |
|TF-IDF | 0.0 | 0.0 | 0.0 |
|Logistic Regression | 0.0 | 0.0 | 0.0 |
|SVM | 0.0 | 0.0 | 0.0 |
|Neural Net | 0.0 | 0.0 | 0.0 |

### Conclusions

After our analysis,

## Quiz

### Can you tell which word cloud comes from troll tweets?

![which_one](pics/nontroll_pic.png)

![which_one2](pics/troll_pic.png)
