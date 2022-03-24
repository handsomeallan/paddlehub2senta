from Senta import Senta
senta = Senta()

test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府', '']

print('######### sentiment prediction, list of sentences ##############')
results = senta.predict(test_data)
for res in results:
    print(res)