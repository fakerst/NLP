# NLP
# Word2Ver-SkipGram：是一种将word转为向量的方法，通过训练将每一个词映射成一个固定长度的向量，所有向量构成一个词向量空间，每一个向量（单词)可以看作是向量空间中的一个点，意思越相近的单词距离越近。其包含两种算法，分别是SkiGram和CBOW，它们的最大区别是SkipGram是通过中心词去预测中心词周围的词，而CBOW是通过周围的词去预测中心词。这里我们实现了SkipGram算法。
# 
