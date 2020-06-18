# Text classification

This code repository covers multi-category classification and multi-label classification, and each classification problem contains a variety of different algorithms.

## Multiclass classification

### Gather Data

Use the THUCNews dataset that contains the text of 740,000 news from the [thuctc](http://thuctc.thunlp.org/).

### Training

```
python -m text_classification.train_ngram_model
```

### Hyperparameter search

```
python -m text_classification.tune_ngram_model
```

### Predicting

```
python -m text_classification.predict_ngram_model
```



## Multi-label classification

### Gather Data

Use the THUCNews dataset that contains the text of 740,000 news from the [thuctc](http://thuctc.thunlp.org/).

### Training

```
python -m text_classification.train_sequence_model
```

or train sequence model with fine-tuned pre-trained embeddings

```
python -m text_classification.train_fine_tuned_sequence_model
```



### Predicting

```
python -m text_classification.predict_sequence_model
```

