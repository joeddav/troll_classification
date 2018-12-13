---
title: SVM
nav_include: 6
notebook: notebooks/SVM.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}


```python
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn import svm
from sklearn import preprocessing
from scipy import sparse
```


## Load Data and Train/Test Split(s)



```python
df=pd.read_json("../data/merged_troll_data.json")
```




```python
df.shape
```





    (332504, 8)





```python
df.sample(5).head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>followers</th>
      <th>following</th>
      <th>retweet</th>
      <th>account_category</th>
      <th>created_at</th>
      <th>troll</th>
      <th>orig_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>307969</th>
      <td>@AIIAmericanGirI @CommonSense1212 @realDonaldT...</td>
      <td>2267</td>
      <td>219</td>
      <td>0</td>
      <td>NonTroll</td>
      <td>2016-11-03 14:35:17</td>
      <td>False</td>
      <td>31589</td>
    </tr>
    <tr>
      <th>135312</th>
      <td>'\|ï¿£ï¿£ï¿£ï¿£ï¿£ï¿£\| \|          33       \...</td>
      <td>629</td>
      <td>364</td>
      <td>1</td>
      <td>LeftTroll</td>
      <td>2016-10-06 13:33:00</td>
      <td>True</td>
      <td>331179</td>
    </tr>
    <tr>
      <th>82234</th>
      <td>'@WOOKIE318 I hope you also pissed off when Cl...</td>
      <td>12321</td>
      <td>9091</td>
      <td>0</td>
      <td>RightTroll</td>
      <td>2016-09-08 01:39:00</td>
      <td>True</td>
      <td>252179</td>
    </tr>
    <tr>
      <th>300364</th>
      <td>RT @HillaryClinton: "Everything I’ve done star...</td>
      <td>3228</td>
      <td>3703</td>
      <td>1</td>
      <td>NonTroll</td>
      <td>2016-11-04 03:02:13</td>
      <td>False</td>
      <td>5521</td>
    </tr>
    <tr>
      <th>37315</th>
      <td>"Understanding will never bring you peace. Tha...</td>
      <td>542</td>
      <td>686</td>
      <td>1</td>
      <td>LeftTroll</td>
      <td>2016-08-07 13:32:00</td>
      <td>True</td>
      <td>303267</td>
    </tr>
  </tbody>
</table>
</div>





```python
ids=pd.read_json("../data/train_test_inds.json")
```




```python
len(ids.random.train)
```





    266003



## Prepare feature matrix

### Isolate matrices



```python
def getxy(ids, feature_cols=['content', 'followers', 'following', 'retweet'], label_col=['troll']):
    return df[feature_cols].iloc[ids], df[label_col].iloc[ids]
```




```python
# random
Xrand_train, yrand_train = getxy(ids.random.train)
Xrand_val, yrand_val = getxy(ids.random.val)
Xrand_test, yrand_test = getxy(ids.random.test)

# temporal
Xtemp_train, ytemp_train = getxy(ids.temporal.train)
Xtemp_val, ytemp_val = getxy(ids.temporal.val)
Xtemp_test, ytemp_test = getxy(ids.temporal.test)
```




```python
Xrand_train.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>followers</th>
      <th>following</th>
      <th>retweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204024</th>
      <td>RT @businessinsider: OBAMA: The press doesn’t ...</td>
      <td>14525</td>
      <td>3311</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45854</th>
      <td>Review: Generation Startup https://t.co/lej8O8...</td>
      <td>3086</td>
      <td>2387</td>
      <td>1</td>
    </tr>
    <tr>
      <th>199686</th>
      <td>RT @Kidrambler: @TomiLahren Vote for Gary John...</td>
      <td>1117</td>
      <td>3742</td>
      <td>1</td>
    </tr>
    <tr>
      <th>115712</th>
      <td>in interpersonal relations with pple who are m...</td>
      <td>936</td>
      <td>582</td>
      <td>1</td>
    </tr>
    <tr>
      <th>245728</th>
      <td>RT @PeterTownsend7: The Real #WarOnWomen  #isi...</td>
      <td>2891</td>
      <td>1615</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





```python
Xrand_train.shape, yrand_train.shape
```





    ((266003, 4), (266003, 1))



### Tokenize content



```python
vocab_size=5000
tokenizer=feature_extraction.text.CountVectorizer(stop_words='english', max_features=vocab_size)
tokenizer=tokenizer.fit(df['content'])
```




```python
Xrand_train_tok=tokenizer.transform(Xrand_train['content'])
Xrand_val_tok=tokenizer.transform(Xrand_val['content'])
Xrand_test_tok=tokenizer.transform(Xrand_test['content'])

Xtemp_train_tok=tokenizer.transform(Xtemp_train['content'])
Xtemp_val_tok=tokenizer.transform(Xtemp_val['content'])
Xtemp_test_tok=tokenizer.transform(Xtemp_test['content'])
```




```python
Xrand_train_tok.shape # token matrix dim = n x vocab_size
```





    (266003, 5000)



### Standardize followers/following



```python
# one for each split
rand_scaler = preprocessing.StandardScaler().fit(Xrand_train[['followers','following']])
temp_scaler = preprocessing.StandardScaler().fit(Xtemp_train[['followers','following']])
```




```python
print('rand means and scales: {}, {}'.format(rand_scaler.mean_, rand_scaler.scale_))
print('temp means and scales: {}, {}'.format(temp_scaler.mean_, rand_scaler.scale_))
```


    rand means and scales: [8154.90645218 3016.03233422], [219679.05451009   7816.52064337]
    temp means and scales: [8757.68069533 3020.22409146], [219679.05451009   7816.52064337]


They are very close. Could probably just use a single one, but I will use both anyways, in case it makes a difference.



```python
col_to_std = ['followers', 'following']
Xrand_train[col_to_std]=rand_scaler.transform(Xrand_train[col_to_std])
Xrand_val[col_to_std]=rand_scaler.transform(Xrand_val[col_to_std])
Xrand_test[col_to_std]=rand_scaler.transform(Xrand_test[col_to_std])

Xtemp_train[col_to_std]=temp_scaler.transform(Xtemp_train[col_to_std])
Xtemp_val[col_to_std]=temp_scaler.transform(Xtemp_val[col_to_std])
Xtemp_test[col_to_std]=temp_scaler.transform(Xtemp_test[col_to_std])
```




```python
Xrand_train[col_to_std].head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>followers</th>
      <th>following</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204024</th>
      <td>0.028997</td>
      <td>0.037736</td>
    </tr>
    <tr>
      <th>45854</th>
      <td>-0.023074</td>
      <td>-0.080475</td>
    </tr>
    <tr>
      <th>199686</th>
      <td>-0.032037</td>
      <td>0.092876</td>
    </tr>
    <tr>
      <th>115712</th>
      <td>-0.032861</td>
      <td>-0.311396</td>
    </tr>
    <tr>
      <th>245728</th>
      <td>-0.023962</td>
      <td>-0.179240</td>
    </tr>
  </tbody>
</table>
</div>



### Binarize the boolean outcome



```python
yrand_train.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>troll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204024</th>
      <td>False</td>
    </tr>
    <tr>
      <th>45854</th>
      <td>True</td>
    </tr>
    <tr>
      <th>199686</th>
      <td>False</td>
    </tr>
    <tr>
      <th>115712</th>
      <td>True</td>
    </tr>
    <tr>
      <th>245728</th>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>





```python
bool_to_bin = lambda x: 1 if x else 0
yrand_train['troll'] = yrand_train['troll'].apply(bool_to_bin)
yrand_train.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>troll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204024</th>
      <td>0</td>
    </tr>
    <tr>
      <th>45854</th>
      <td>1</td>
    </tr>
    <tr>
      <th>199686</th>
      <td>0</td>
    </tr>
    <tr>
      <th>115712</th>
      <td>1</td>
    </tr>
    <tr>
      <th>245728</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
yrand_val['troll'] = yrand_val['troll'].apply(bool_to_bin)
yrand_test['troll'] = yrand_test['troll'].apply(bool_to_bin)

ytemp_train['troll'] = ytemp_train['troll'].apply(bool_to_bin)
ytemp_val['troll'] = ytemp_val['troll'].apply(bool_to_bin)
ytemp_test['troll'] = ytemp_test['troll'].apply(bool_to_bin)
```


### Concatenate features



```python
def concatenate_features(tok_matrix, data_df):
    """ concatenate the tokenized matrix (scipy.sparse) with other features """
    sparse_cols = sparse.csr_matrix(data_df[['followers', 'following', 'retweet']])
    combined = sparse.hstack([tok_matrix, sparse_cols])
    return combined
```




```python
Xrand_train_combined = concatenate_features(Xrand_train_tok, Xrand_train)
Xrand_val_combined = concatenate_features(Xrand_val_tok, Xrand_val)
Xrand_test_combined = concatenate_features(Xrand_test_tok, Xrand_test)

Xtemp_train_combined = concatenate_features(Xtemp_train_tok, Xtemp_train)
Xtemp_val_combined = concatenate_features(Xtemp_val_tok, Xtemp_val)
Xtemp_test_combined = concatenate_features(Xtemp_test_tok, Xtemp_test)
```


## Train the model(s)

### Using only text



```python
# random split
svm_model = svm.SVC().fit(Xrand_train_tok, yrand_train['troll'])
```




```python
svm_model.score(Xrand_train_tok, yrand_train['troll'])
```





    0.8563023725296331





```python
svm_model.score(Xrand_val_tok, yrand_val['troll'])
```





    0.8545563909774436





```python
svm_model.score(Xrand_test_tok, yrand_test['troll'])
```





    0.8563652220985835





```python
# temporal split
svm_temp = svm.SVC().fit(Xtemp_train_tok, ytemp_train['troll'])
```




```python
svm_temp.score(Xtemp_val_tok, ytemp_val['troll'])
```





    0.8595187969924812





```python
svm_temp.score(Xtemp_test_tok, ytemp_test['troll'])
```





    0.8649664671739196



### Using all features



```python
# random split
svm_rand_all = svm.SVC().fit(Xrand_train_combined, yrand_train['troll'])
```




```python
svm_rand_all.score(Xrand_train_combined, yrand_train['troll'])
```





    0.9306173238647685





```python
svm_rand_all.score(Xrand_test_combined, yrand_test['troll'])
```





    0.928693873868455





```python
# temporal split
svm_temp_all = svm.SVC().fit(Xtemp_train_combined, ytemp_train['troll'])
```




```python
svm_temp_all.score(Xtemp_train_combined, ytemp_train['troll'])
```




```python
svm_temp_all.score(Xtemp_test_combined, ytemp_test['troll'])
```





    0.8745902378875823
