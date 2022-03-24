from senta import Senta

import Senta

my_senta = Senta('infer_model')

# 获取目前支持的情感预训练模型, 我们开放了以ERNIE 1.0 large(中文)、ERNIE 2.0 large(英文)和RoBERTa large(英文)作为初始化的SKEP模型
print(my_senta.get_support_model()) # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]

# 获取目前支持的预测任务
print(my_senta.get_support_task()) # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]

# 选择是否使用gpu
use_cuda = False # 设置True or False

# 预测英文句子级情感分类任务（基于SKEP-ERNIE2.0模型）
my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
texts = ["a sometimes tedious film ."]
result = my_senta.predict(texts)
print(result)

test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府', '']

print('######### sentiment prediction, list of sentences ##############')
results = my_senta.predict(test_data)
for res in results:
    print(res)