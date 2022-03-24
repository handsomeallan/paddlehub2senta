# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import paddlehub as hub
def getSentement():
    senta = hub.Module(name="senta_bilstm")
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    results = senta.sentiment_classify(texts=test_text, use_gpu=False, batch_size=1)

    for result in results:
        print(result['text'])
        print(result['sentiment_label'])
        print(result['sentiment_key'])
        print(result['positive_probs'])
        print(result['negative_probs'])

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    getSentement()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
