import requests
import json
import time
import re
 
url = 'https://api.openai.com/v1/chat/completions'
headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

def main():
  f = open('./datasets/gen_query.json', 'r')
  fw = open('./datasets/gen_keyword.json', 'w')

  for line in f:
      line = json.loads(line)
      text = line['text']
      query_dict = line['query_dict']
      for query_type in query_dict.keys():
        if not query_dict[query_type]:
          continue
        
        query = query_dict[query_type]
        line['query_type'] = query_type
        line['query'] = query
        content = f'''
          Assuming you are a data generator, please generate coarse-grained keywords and fine-grained keywords according to the query and context as required. You can refer to the given examples.

          ###Requirements###
          1. Both coarse-grained keywords and fine-grained keywords are a text fragment of a given context;
          2. The output can only be a json dictionary, which can be parsed by json.loads();
          3. The output dictionary contains coarse-grained keywords and fine-grained keyword lists, where the elements of the fine-grained keyword list are also lists (each sublist represents the result of splitting the same information point into clauses);
          4. Coarse-grained keywords are generally entities, which are one or more entities that best represent the topic of the given context and query;
          5. Fine-grained keywords should be divided according to the query, and each sublist corresponds to an information point, that is, the key point of answering the question. When the queries are different, the grouping of fine-grained keywords should be different;
          6. If there is no suitable keyword in the context, the corresponding output should be an empty list.

          ###Examples###
          Query: 为什么煤炭的稳定供应对下游用煤企业至关重要？
          Context: 笔者认为有几点需要关注。\n供销稳定是基础。煤炭作为国家工业粮食，与下游电力、冶\n金、化工、建材、水泥等行业的发展息息相关。煤炭稳定供应是\n下游用煤企业的压舱石和稳定器。无论市场如何变化，煤炭生产\n和销售企业都应在追求经济效益最大化的同时，履行好社会责\n任，做好煤炭市场供应工作。煤炭生产企业要加大安全生产管控\n力度，确保安全稳定生产。运销企业要把握产销平衡的原则，利\n用好销售渠道进行高效销售，同时布局港口、物流节点等煤炭物\n流储备基地，发挥储备基地蓄水池作用，实现煤炭淡旺季销售科\n学调度，确保市场供应行稳致远。\n运力稳定是保障。铁路是煤炭的主要运输方式，更是煤炭市\n场稳定供应的重要保障。运销企业要和铁路部门同频共振，在产\n能释放、销售增量、运力协调等方面加强与铁路运输部门的合作，\n根据市场布局和销售情况及时调整运力部署，实现稳定装车快进\n快出。铁路部门要根据运销企业的供应能力和市场布局提供稳定\n的运力，确保运力运输高效配置，为供运高质量运行创造条件。\n履行合同是关键。煤炭中长协合同既是国家稳定煤炭市场供\n应的管理机制，更是供运需三方的市场契约。在交易环节中，重\n合同守信誉是市场的准则，三方要把履行合同作为检验各自企业\n诚信经营的试金石。销售方和终端用户要坚持年度大合同月度小\n补充的灵活方式，实现供应方及时足量供应，用户方按合同按时\n间及时接卸；销售方和运输方要依照运力合同坚持互惠共赢全力-3-高效承运，通过各自发力寻找契合点，履职尽责协作顺畅，实现\n合作高质高效。\n严格监管是手段。稳固的煤炭市场供运需关系不仅需要合同\n的约束，更需要政府相关部门的严格监管，才能实现供运需关系\n的良性循环发展。监管部门须加大合同履行的检查力度，实行动\n态考评和惩戒淘汰机制，及时协调处理履行过程中存在的问题，\n为三方履约提供政策指导和精细服务。对履约到位的给予鼓励支\n持，对不能履约的及时约谈督导，把属于市场的交还给市场，把\n属于政府监管的严格约束。进一步用活监管监督手段，让“有形\n的手”和“无形的手”紧握在一起，促进构建稳固的煤炭供运需\n关系，为三方企业的高质量发展保驾护航。\n集团要闻\n邓伟、刘万波一行到四川高兴煤炭储备基地\n开展调研工作\n2020年11月5日，省能源局煤炭处处长邓伟、川煤集团临\n时负责人、党委副书记、副董事长、总经理刘万波等一行到四川\n省兴铁多式联运有限公司（以下简称“兴铁公司”）调研储备煤\n基地建设情况。川煤集团广能公司董事长肖前昌、国新联程总经\n理赵麒麟等有关领导陪同调研。-4-调研会上，邓伟强调要充分认识到高兴储备煤基地建设的重\n要性，必将抢抓机遇，政府将全力支持项目建设。
          Answer: {{'coarse_keywords':[
                  "煤炭"
              ],
              'fine_keywords':[
                ["供销稳定是基础"],
                ["煤炭作为国家工业粮食", "与下游电力、冶金、化工、建材、水泥等行业的发展息息相关"],
                ["煤炭稳定供应是下游用煤企业的压舱石和稳定器"],
                ["煤炭生产企业", "加大安全生产管控力度", "确保安全稳定生产"],
                ["运销企业", "把握产销平衡的原则", "利用好销售渠道进行高效销售", "布局港口、物流节点等煤炭物流储备基地"],
                ["运力稳定是保障"],
                ["铁路是煤炭的主要运输方式", "煤炭市场稳定供应的重要保障"],
                ["运销企业", "与铁路运输部门的合作", "根据市场布局和销售情况及时调整运力部署"]
              ]
              }}

          Query: AI技术经历了三代，分别是哪三代呢？分别有哪些代表公司？
          Context: 产品技术背景– AI技术趋势；第一代：符号AI，符号模型 规则模型 感知机器；第二代：感知智能，大数据驱动的统计学习方法初步实现了针对文本、图像、语音等的感知与识别；第三代：认知智能；张钹院士提出第三代人工智能雏形，DARPA 2018年发布AI Next计划。核心思路是推进数据统计与知识推理融合的计算；与脑认知机理融合的计算。
          Answer: {{'coarse_keywords':[
                "AI技术"
              ],
              'fine_keywords':[
                ["产品技术背景– AI技术趋势"],
                ["第一代：符号AI", "符号模型 规则模型 感知机器"],
                ["第二代：感知智能", "大数据驱动的统计学习方法初步实现了针对文本、图像、语音等的感知与识别"],
                ["第三代：认知智能"],
                ["张钹院士提出第三代人工智能雏形", "DARPA 2018年发布AI Next计划。核心思路是推进数据统计与知识推理融合的计算"],
                ["与脑认知机理融合的计算。"]
              ]
          }}

          Query: {query}
          Context: {text}
          Answer:
      '''
        data = {
                  "model": "gpt-4o",
                  "messages": [{"role": "user", "content": content}]
                }
        try:
            response = requests.post(url, headers=headers, json=data)
            res = json.loads(response.text)
            ans = res['data']['response']['choices'][0]['message']['content']
            matches = re.findall(r"\{.*?\}", ans.replace('\n', ''))
            answer_dict = eval(matches[0])  
            line['fine_keywords'] = answer_dict['fine_keywords']
            line['coarse_keywords'] = answer_dict['fine_keywords']
            fw.write(json.dumps(line, ensure_ascii=False)+'\n')
        except Exception as e:
            print(f"{e}")
    
if __name__ == '__main__':
  main()
