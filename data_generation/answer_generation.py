import requests
import json
 
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
          Assume you are an intelligent assistant, please answer the query based on the context provided.

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
            line['reference_answer'] = ans
            fw.write(json.dumps(line, ensure_ascii=False)+'\n')
        except Exception as e:
            print(f"{e}")
    
if __name__ == '__main__':
  main()
