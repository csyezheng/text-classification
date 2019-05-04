import load_data
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import numpy as np


def predict_ngram_model(content, learning_rate=1e-3):
    """predict n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

    tokenize = load_data.text_segmentation(content)

    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    selector = pickle.load(open("selector.pickle", "rb"))
    x_predict = vectorizer.transform([tokenize])
    x_predict = selector.transform(x_predict)

    model = load_model('cnews_mlp_model.h5')
    loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    predict = model.predict(x_predict)
    cat_id = np.argmax(predict, axis=1)[0]
    categories, cat_to_id = load_data.read_category()
    result = categories[cat_id]
    print(result)



if __name__ == '__main__':
    # content = "专家预测最后9战6胜3负 火箭闯入前八概率只三成新浪体育讯北京时间3月29日消息，美国网站ESPN专家约翰-霍林格给出了他自己的季后赛出线预测，根据他的预测，主要竞争西部季后赛席位的几支球队晋级概率如下：开拓者96.3%、黄蜂93.0%、灰熊87.5%、火箭22.6%、太阳0.6%。换句话说，霍林格认为火箭晋级季后赛的希望已经不足三成了。霍林格的这项预测是基于各队目前的状况，以及随后的赛程。霍林格认为还有9场比赛没打的火箭，最佳战绩可能是9胜0负，而最差战绩有可能是1胜8负。最可能的战绩是44胜38负，也就是在这9场比赛中取得6胜3负的成绩。而霍林格预测火箭的竞争对手在常规赛结束后最可能出现的战绩分别是开拓者47胜35负，黄蜂46胜36负，灰熊46胜36负，火箭将以两场的差距无缘季后赛。应该说，这项分析还是合情合理的，除非有奇迹出现，否则火箭晋级季后赛的希望确实只有不到三成了。该项数据还给出了火箭打进总决赛的概率，是1.9%，而夺取总冠军的概率只有0.6%。当然，这些微小的概率只是理论上的，如果火箭连季后赛都打不进去，这些概率也只能成为大家饭后的谈资。再来看看其他球队的情况，根据这项数据预测，西部晋级总决赛概率最高的是湖人，达到23.7%，随后依次是掘金、马刺和小牛。东部晋级总决赛概率最高的是公牛，达到49.3%，随后依次是热火、凯尔特人和魔术。在夺冠概率上，公牛一马当先，达到了31.1%，湖人和热火平分秋色，都是11.6%。显然，在霍林格眼中，东部异军突起的公牛已经成为了夺冠第一热门，罗斯将带领他的球队走上芝加哥复兴之路，而热火和湖人依然是他们最强劲的竞争者。至于年纪稍大的凯尔特人、马刺，则不被霍林格所看好。(肥仔)"
    content = "我一直呼吁，各类销售机构都能够去服务老百姓。我们希望在现有的基金销售生态上，建立个人养老账户。个人养老的钱实际上不像平常买基金随时能够申赎的，如果通过税收的方式做了这种安排，实际上是把钱要封闭20年甚至30年。所以未来如何让销售机构跟中登的平台连接，能够把资金封闭运作，这是我们个人税收递延的核心，目前中登跟数十家基金公司、银行和券商、以及互联网平台做了联测，我们做好了账户封闭运营的基础设施平台"
    predict_ngram_model(content)