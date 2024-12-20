import os
import requests
import json
import time
import re
import tiktoken

url = 'https://api.openai.com/v1/chat/completions'
headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

def encode_string_by_tiktoken(content, model_name = "gpt-4o"):
    ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def split_paragraph(paragraph, max_length=1200):
    sentences = re.split(r'([。！？])', paragraph)
    if sentences[-1] == '':
        sentences = sentences[:-1]
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)] + [sentences[-1]]
    result = []
    current_chunk = ''
    for sentence in sentences:
        if len(encode_string_by_tiktoken(current_chunk)) + len(encode_string_by_tiktoken(sentence)) > max_length:
            result.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
    if current_chunk:
        result.append(current_chunk.strip())
    return result


def main():
    fw = open('./datasets/gen_query.json', 'w')
    dir_path = './datasets/docdata/offline_parse_li_native_pdf'

    for path in os.listdir(dir_path):
        if not path.endswith('.pdf.document'):
            continue

        doc = json.load(open(os.path.join(dir_path, path)))
        all_text = ''
        for page in doc:
            all_text += page['text']
    
        context = split_paragraph(all_text.strip())

        for text in context:
            line = {}
            line['metadata'] = page['metadata']
            line['text'] = text
            
            content = f'''
                Assuming you are a data generator, please construct four types of queries as required for each given context. You can refer to the given examples.

                ###Requirements###
                1. The queries should be in line with human style and independent of each other;
                2. The queries should be clear, specific and detailed, without vague references such as pronouns (such as "this", "it", etc);
                3. The queries should be able to derive answers from the given content, and the corresponding content can be retrieved through the query;
                4. If a certain type cannot obtain a query that meets the above requirements, the output of the corresponding query type is empty.
                5. The output can only be a json dictionary, which can be parsed by json.loads(). 

                ###Query Type Definition###
                Factual: Seeking specific, clear facts or evidence. Example: When was the Beijing Olympics held? Where is the capital of the United States?
                Analytical: Seeking analytical explanations or summaries of specific concepts, terms, or phenomena. Example: Why is the earth warming? What are the advantages of renewable energy?
                Comparative: Seeking comparisons of information in different dimensions. Example: Which is more developed, Japan or South Korea? What are the differences between Western medicine and traditional Chinese medicine in treating chronic diseases?
                Tutorial: Seeking the steps to perform a task or process. Example: How to get a driver's license? What are the steps to install TensorFlow?

                ###Output Format###
                {{"Factual":"", "Analytical":"", "Comparative":"", "Tutorial":""}}

                ###Examples###
                Context: 公司简介   杭州天宽科技有限公司（以下简称：天宽）成立于 2007 年，总部位于杭州，是国内知名的高科技企业。\n天宽长期聚焦于智能电网、云计算和军工行业的信息化建设等领域，为客户提供包括云基础架构的建设与维\n护、下一代通信保障与防护系统、移动互联网行业化应用等专属解决方案。  \n        天宽一直以客户需求为导向，凭借先进的技术理念和强大的技术实力，赢得了客户的尊重与信赖。目\n前与国家电网、中国移动通信集团公司、欧洲 O2移动通信集团和阿里巴巴集团等世界 500强企业保持着长期\n友好的商业合作关系。在国内，公司的经营范围已覆盖全国近 20个省和直辖市；同时天宽在德国和西班牙实\n现了持续的技术服务收入，为公司业务在海外市场的拓展奠定了坚实的基础。  \n       天宽坚持技术创新，被评为国家级高新技术企业和浙江省软件企业，获得计算机系统集成二级资质、杭\n州市名牌产品、杭州市著名商标等多项殊荣，并取得多项国家专利、软件产品登记和著作权。天宽研发机构\n健全，其中研发中心已通过 CMMI3 级认证，并被评为杭州市级高新技术研发中心。  \n未来，天宽将继续提升品牌影响力，致力于推动全球信息服务业的发展，成为国际一流的信息技术专属解决\n方案提供商 。
                Answer: {{
                    "Factual": "杭州天宽科技有限公司成立于哪一年？",
                    "Analytical": "杭州天宽科技有限公司在国内的市场覆盖情况是怎样的？",
                    "Comparative": "",
                    "Tutorial": ""
                }}

                Context: {text}
                Answer:
            '''
            data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": content}],
                    }
         
            try:
                response = requests.post(url, headers=headers, json=data)
                res = json.loads(response.text)
                ans = res['data']['response']['choices'][0]['message']['content']
                matches = re.findall(r"\{.*?\}", ans.replace('\n', ''))
                answer_dict = eval(matches[0])  
                line['query_dict'] = answer_dict
                fw.write(json.dumps(line, ensure_ascii=False)+'\n')
            except Exception as e:
                print(f"{e}")
        
if __name__ == '__main__':
    main()