- aspect分类的种类的为：
Counter({'整体': 2822, '使用体验': 1042, '功效': 726, '价格': 696, '物流': 517, '气味': 225, '包装': 195, '真伪': 161, '服务': 86, '其他': 65, '成分': 61, '尺寸': 24, '新鲜度': 13})

- 序列化标注的标注方法
采取BIO的标注方法
AspectTerms: 追加尾椎at
OpinionTerms： 追加尾椎ot

@: 应为单词
&：代表数字

- 参考项目：https://github.com/macanv/BERT-BiLSTM-CRF-NER

- process_data.py：准备运行数据

- train_helper.py： 配置运行的文件参数以及模型参数

- run.py: 运行接口