# 基于keras的文本分类实现


| Model         | Description                                                                                                                                                       | Reference                                                                                                                                                                    |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Simple RNN    | 简单的RNN实现，直接将GRU的输出传递至使用softmax激活的全连接层。                                                                                                   |
| Simple Bi-RNN | 与Simple RNN类似，只不过是将单向的GRU替换为了双向GRU。                                                                                                            |
| Simple CNN    | 使用CNN实现的简单文本分类模型，直接RNN替换为了两个堆叠的核大小为7的1维卷积层，并使用了一个1维全局最大池化，在整个时间步上进行池化，将变长输入序列转换为定长向量。 |
| Text CNN      | 将三个卷积层的输出进行拼接                                                                                                                                        |
| Text RCNN     | 使用一个双向RNN代替卷积，并将其输出与嵌入层的输出进行拼接，通过一个带有tanh的线性变换来实现参特征融合                                                             | [Lai S, Xu L, Liu K, et al. Recurrent convolutional neural networks for text classification\[C\]//Twenty-ninth AAAI conference on artificial intelligence. 2015.][text-rcnn] |

[text-rcnn]: https://www.deeplearningitalia.com/wp-content/uploads/2018/03/Recurrent-Convolutional-Neural-Networks-for-Text-Classification.pdf


## 快速开始

### 创建项目

```sh
python -m text_classification --project demo init
```

`data`目录用于存放训练数据集，目前训练数据集仅支持`.pickle.gz`格式的文件，其是一个3元组或4元组：

- `train_data` 训练数据，元组 `(x_train, y_train)`
- `validation_data` （可选）验证数据，元组 `(x_val, y_val)`。如果验证集不存在，则默认随机使用测试集的前50%作为验证集
- `test_data` 测试数据，元组 `(x_test, y_test)`
- `data_dict` 字典，至少包含 `word_index`, `classes`
  - `word_index` 单词索引字典
  - `classes` 文本类别列表，字符串表示

英文分类配置
```yaml
lang: en
data:
  dataset: imdb_mini.pickle.gz
```

中文分类配置
```yaml
lang: cn
data:
  dataset: zh_hotel_review.pickle.gz
```

### 训练模型

```sh
python -m text_classification --project=examples/simple train --model=SimpleRNN
```

### 模型预测

```sh
python -m text_classification --project=examples/simple predict 'this is a good moive, amazing!'
```