## Step 1: Gather Data

Source data from public data set on BBC news articles. 

Its original source.its original source:  <http://mlg.ucd.ie/datasets/bbc.html>

Cleaned up version: <https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv>



## Step 2: Explore Data

Is the number of articles in each category roughly equal?

If our dataset were imbalanced, we would need to carefully configure our model or artificially balance the dataset, for example by [**undersampling** or **oversampling**](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) each class.



## Step 2.5: Choose a Model

use a simple multi-layer perceptron (MLP) model



## Step 3: Prepare Data

### extracting features

To further analyze our dataset, we need to transform each article's text to a feature vector, a list of numerical values representing some of the text’s characteristics. 

* one-hot vector
* Co-occurrence Matrix (SVD Based Methods)

* ##### word2vec 

  * ##### continuous bag of words model

    a model where for each document, an article in our case, the presence (and often the **frequency**) of words is taken into consideration, but the order in which they occur is ignored.

  * skip-gram

* Term Frequency, Inverse Document Frequency (**tf-idf**)

  This statistic represents words’ importance in each document.



## Step 4: Build, Train, and Evaluate Model



Select and Train a Model

We don't want to touch the test set until we are ready to launch a model you are confident about, so we need to use part of the training set for training, and part for validation.



**Train on 50000 samples, validate on 5000 samples, evaluate on 10000 samples**



### Performance Measures

#### Measuring Accuracy Using Cross-Validation

#### Confusion Matrix

#### Precision and Recall

#### Precision / Recall Tradeoff

#### The ROC Curve

we must carefully choose the right metric based on the task we are trying to solve. Here, we are dealing with a multi-class classification task (trying to assign one label to each document, out of a number of labels). Given the relative balance of our dataset, **accuracy** does seem like a good metric to optimize for. If one of the labels was more important than the others, we would then weight it higher, or focus on its own results. For imbalanced datasets such as the one described above, it is common to look at the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), and optimize the Area Under the Curve (**ROC AUC**).



```
Epoch 1/1000
 - 82s - loss: 4.0544 - acc: 0.1408 - val_loss: 2.2060 - val_acc: 0.2986
Epoch 2/1000
 - 82s - loss: 1.4355 - acc: 0.6531 - val_loss: 0.6545 - val_acc: 0.9208
Epoch 3/1000
 - 82s - loss: 0.4620 - acc: 0.9093 - val_loss: 0.2935 - val_acc: 0.9484
Epoch 4/1000
 - 82s - loss: 0.2653 - acc: 0.9431 - val_loss: 0.2181 - val_acc: 0.9540
Epoch 5/1000
 - 80s - loss: 0.1956 - acc: 0.9527 - val_loss: 0.1909 - val_acc: 0.9556
Epoch 6/1000
 - 81s - loss: 0.1565 - acc: 0.9618 - val_loss: 0.1716 - val_acc: 0.9588
Epoch 7/1000
 - 82s - loss: 0.1353 - acc: 0.9651 - val_loss: 0.1641 - val_acc: 0.9586
Epoch 8/1000
 - 82s - loss: 0.1178 - acc: 0.9695 - val_loss: 0.1547 - val_acc: 0.9616
Epoch 9/1000
 - 82s - loss: 0.1024 - acc: 0.9731 - val_loss: 0.1535 - val_acc: 0.9616
Epoch 10/1000
 - 81s - loss: 0.0926 - acc: 0.9759 - val_loss: 0.1493 - val_acc: 0.9614
Epoch 11/1000
 - 82s - loss: 0.0818 - acc: 0.9785 - val_loss: 0.1514 - val_acc: 0.9604
Epoch 12/1000
 - 82s - loss: 0.0729 - acc: 0.9813 - val_loss: 0.1527 - val_acc: 0.9590
Validation accuracy: 0.9589999914169312, loss: 0.15271664148569106
10000/10000 [==============================] - 6s 587us/sample - loss: 0.1445 - acc: 0.9614

Loss: 0.14, Accuracy: 96.14%

```



## Step 5: Tune Hyperparameters

## Step 6: Deploy Model

#### predict

```
专家预测最后9战6胜3负 火箭闯入前八概率只三成新浪体育讯北京时间3月29日消息，美国网站ESPN专家约翰-霍林格给出了他自己的季后赛出线预测，根据他的预测，主要竞争西部季后赛席位的几支球队晋级概率如下：开拓者96.3%、黄蜂93.0%、灰熊87.5%、火箭22.6%、太阳0.6%。换句话说，霍林格认为火箭晋级季后赛的希望已经不足三成了。霍林格的这项预测是基于各队目前的状况，以及随后的赛程。霍林格认为还有9场比赛没打的火箭，最佳战绩可能是9胜0负，而最差战绩有可能是1胜8负。最可能的战绩是44胜38负，也就是在这9场比赛中取得6胜3负的成绩。而霍林格预测火箭的竞争对手在常规赛结束后最可能出现的战绩分别是开拓者47胜35负，黄蜂46胜36负，灰熊46胜36负，火箭将以两场的差距无缘季后赛。应该说，这项分析还是合情合理的，除非有奇迹出现，否则火箭晋级季后赛的希望确实只有不到三成了。该项数据还给出了火箭打进总决赛的概率，是1.9%，而夺取总冠军的概率只有0.6%。当然，这些微小的概率只是理论上的，如果火箭连季后赛都打不进去，这些概率也只能成为大家饭后的谈资。再来看看其他球队的情况，根据这项数据预测，西部晋级总决赛概率最高的是湖人，达到23.7%，随后依次是掘金、马刺和小牛。东部晋级总决赛概率最高的是公牛，达到49.3%，随后依次是热火、凯尔特人和魔术。在夺冠概率上，公牛一马当先，达到了31.1%，湖人和热火平分秋色，都是11.6%。显然，在霍林格眼中，东部异军突起的公牛已经成为了夺冠第一热门，罗斯将带领他的球队走上芝加哥复兴之路，而热火和湖人依然是他们最强劲的竞争者。至于年纪稍大的凯尔特人、马刺，则不被霍林格所看好。(肥仔)
```

The predict script classifies this text as `体育`

```
我一直呼吁，各类销售机构都能够去服务老百姓。我们希望在现有的基金销售生态上，建立个人养老账户。个人养老的钱实际上不像平常买基金随时能够申赎的，如果通过税收的方式做了这种安排，实际上是把钱要封闭20年甚至30年。所以未来如何让销售机构跟中登的平台连接，能够把资金封闭运作，这是我们个人税收递延的核心，目前中登跟数十家基金公司、银行和券商、以及互联网平台做了联测，我们做好了账户封闭运营的基础设施平台
```

The predict script classifies this text as `财经`



