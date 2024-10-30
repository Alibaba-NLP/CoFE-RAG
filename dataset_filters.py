# 创建一个全局字典来存储所有注册的函数
filters_registry = {}

def filter(func):
    """
    装饰器，将函数自动注册到filters_registry字典中。
    """
    # 将函数添加到全局字典中，键是函数的名称，值是函数本身
    filters_registry[func.__name__] = func
    return func

@filter
def no_filter(example):
    return True

@filter
def bussiness_dataset_knowledge_only(example):
    print(example)
    meta_info = example['meta_info']
    if meta_info['是否废弃'] == '否' and meta_info['query能否在url中找出答案'] == '是' and meta_info['range'] == '知识库':
        return True
    return False