import json
import os

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core import Document

def nodefile2node(input_file):
    nodes = [TextNode.from_dict(doc) for doc in json.load(open(input_file, 'r'))]
    return nodes

def onlchunkfile2node(input_file):
    content_json = json.load(open(input_file, 'r'))
    nodes = []
    for data in content_json:
        node = TextNode(text=data['title'] + data.get('hier_title', '') + data['content'], file_name=input_file)
        nodes.append(node)
        if len(nodes) > 1:
            nodes[-1].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[-2].node_id
            )
            nodes[-2].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[-1].node_id
            )
    return nodes


def transform_idp2markdown(response_json: dict) -> str:
    # 初始化Markdown字符串
    markdown_text = ""
    if 'layouts' in response_json:
        response_json = response_json['layouts']

    # 遍历layouts数组
    for layout in response_json:
        if layout is None:
            continue
        if not 'subType' in layout:
            layout['subType'] = 'para'
        # 根据类型设置Markdown格式
        if layout["type"] == "title": 
            # 文档标题使用一级标题
            markdown_text += "\n\n\n" + layout["text"] + '\n'
        else:
            # 正文使用段落格式
            markdown_text += layout["text"] + "\n"
    return markdown_text

def documentfile2document(input_file):
    documents = [Document.from_dict(doc) for doc in json.load(open(input_file, 'r'))]
    return documents



def text2document(input_file):
    text = open(input_file, 'r').read()
    metadata = {"file_name": input_file}
    documents = [Document(text=text, metadata=metadata)]
    return documents