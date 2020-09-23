# 基于keras的中文文本分类实现


| Model      | Description                                                                                                                                                       | Reference                                                                                                                                                                    |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Simple RNN | 简单的RNN实现，直接将双向GRU的输出传递至使用softmax激活的全连接层。                                                                                               |
| Simple CNN | 使用CNN实现的简单文本分类模型，直接RNN替换为了两个堆叠的核大小为7的1维卷积层，并使用了一个1维全局最大池化，在整个时间步上进行池化，将变长输入序列转换为定长向量。 |
| Text CNN   | 将三个卷积层的输出进行拼接                                                                                                                                        |
| Text RCNN  | 使用一个双向RNN代替卷积，并将其输出与嵌入层的输出进行拼接，通过一个带有tanh的线性变换来实现参特征融合                                                             | [Lai S, Xu L, Liu K, et al. Recurrent convolutional neural networks for text classification\[C\]//Twenty-ninth AAAI conference on artificial intelligence. 2015.][text-rcnn] |

[text-rcnn]: https://www.deeplearningitalia.com/wp-content/uploads/2018/03/Recurrent-Convolutional-Neural-Networks-for-Text-Classification.pdf


## 快速开始

```
$ python -m text_classification --help
usage: __main__.py [-h] [--data DATA] [--data-field-label DATA_FIELD_LABEL]
                   [--data-field-text DATA_FIELD_TEXT] [--models MODELS]
                   [--load-model LOAD_MODEL] [--model MODEL] [--train]
                   [texts [texts ...]]

positional arguments:
  texts

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data path
  --data-field-label DATA_FIELD_LABEL
  --data-field-text DATA_FIELD_TEXT
  --models MODELS       models directory
  --load-model LOAD_MODEL
                        the path of the model to be loaded
  --model MODEL         model name, used in training phase
  --train               train model
```

在开始之前，如果编辑器没有自动将本项目所在的位置加入`PYTHONPATH`，可能需要手动更新`PYTHONPATH`：

```sh
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 查看支持的模型

```
$ python -m text_classification --ls

['text_rcnn', 'text_cnn', 'simple_rnn', 'simple_cnn']
```

### 模型训练

```sh
python -m text_classification --data data/zh_hotel_review.csv --data-field-text review --model text_cnn --train
```

训练模型需要提供以下参数：

- `data` 训练数据的路径，目前支持`.csv`, `.xls`, `.xlsx`
  - `data-field-label` 原始数据，对应标签列的列名，默认为`label`
  - `data-field-text` 原始数据，对应文本列的列名，默认为`text`
- `models` 存放所有模型的目录，所有模型按照`model-[iso time format]`格式存放于此路径下。默认会使用当前路径下的`models`目录。
- `model` 训练所使用的模型的名称

### 数据预测

```sh
python -m text_classification '价格比比较不错的酒店。这次免费升级了，感谢前台服务员。房子还好，地毯是新的，比上次的好些。' '非常差的服务！！！！你要继续误导消费者吗？！'

Predict results:
>> ['1', '0']
```

数据预测需要提供以下参数：
- `models` 存放所有模型的目录，如果没有提供`load-model`参数，将会自动调用最新的模型。
- `load-model` 将要加载的模型路径

### Web服务

```sh
python -m text_classification --server
```

接口支持如下：

| Name         | Router        | Method | Comment                                        |
| ------------ | ------------- | ------ | ---------------------------------------------- |
| 文本分类预测 | `/predict`    | POST   | 对文本序列进行分类，参数为JSON格式的字符串数组 |
| 分类标签     | `/classes`    | GET    | 获取当前模型所支持的分类标签                   |
| 单词映射字典 | `/word_index` | GET    | 获取当前模型所使用的单词映射字典               |



## 变更历史


### 2020.09.23

1. 对分类工具增加了基于Flask的Web API，当前不支持训练。
2. （实验性）增加Docker支持

### 2020.09.22

1. 对整体代码进行了简化，删除了配置文件模式，对命令行接口进行了重构，支持通过`--stdin`读取标准输入
2. 修复由于输入序列过短导致卷积核无法工作的BUG，在预测阶段对所有输入的长度强制重置为256