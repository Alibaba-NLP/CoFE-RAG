import re

def clean_text(text):
    # 去除所有中英文标点符号以及空格
    text = re.sub(r'[^\w\u4e00-\u9fa5]+', '', text)
    text = text.replace('一一', '一')
    return text
