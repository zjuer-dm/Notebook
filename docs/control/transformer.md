# Transformer

Transformer是一种用于自然语言处理（NLP）和其他序列到序列（sequence-to-sequence）任务的深度学习模型架构，它在2017年由Vaswani等人首次提出。Transformer架构引入了自注意力机制（self-attention mechanism），这是一个关键的创新，使其在处理序列数据时表现出色。


## 发展
![alt text](<src/Screenshot 2024-12-10 at 14.56.36.png>)

Transformer最初应用于序列到序列的自动回归任务，特别是在机器翻译中。与传统的递归神经网络（RNN）和长短期记忆网络（LSTM）相比，Transformer通过多头注意力机制和逐位置前馈网络，完全放弃了递归和卷积，从而显著提高了计算效率和模型性能

## 框架
![alt text](<src/Screenshot 2024-12-10 at 15.41.39.png>)

编码器和解码器：Transformer通常包括一个编码器用于处理输入序列和一个解码器用于生成输出序列，这使其适用于序列到序列的任务，如机器翻译。

堆叠层（Stacked Layers）：Transformer通常由多个相同的编码器和解码器层堆叠而成。这些堆叠的层有助于模型学习复杂的特征表示和语义。

## 注意力
自注意力机制（Self-Attention）：这是Transformer的核心概念之一，它使模型能够同时考虑输入序列中的所有位置，而不是像循环神经网络（RNN）或卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中的不同部分来赋予不同的注意权重，从而更好地捕捉语义关系。

多头注意力（Multi-Head Attention）：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力允许模型并行处理不同的信息子空间。

![alt text](<src/Screenshot 2024-12-10 at 19.01.12.png>)

自注意力的计算：从每个编码器的输入向量（每个单词的词向量，即Embedding，可以是任意形式的词向量，比如说word2vec，GloVe，one-hot编码）

自注意力的计算：从每个编码器的输入向量（每个单词的词向量，即Embedding，可以是任意形式的词向量，比如说word2vec，GloVe，one-hot编码）
中生成三个向量，即查询向量、键向量和一个值向量。（这三个向量是通过词嵌入与三个权重矩阵即Q、K、V,相乘后创建出来的）新向量在维度上往往比词嵌入向量更低。（512->64）

更一般的，将以上所得到的查询向量、键向量、值向量组合起来就可以得到三个向量矩阵Query、Keys、Values。