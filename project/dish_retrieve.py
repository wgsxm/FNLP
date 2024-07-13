import json
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import torch

test_data = json.load(open('./dish-name-recognition/test_recipe.json', 'r', encoding='utf-8'))
train_data = json.load(open('./dish-name-recognition/train.json', 'r', encoding='utf-8'))
os.makedirs('./log_retrieve', exist_ok=True)
client = OpenAI(api_key="sk-7d2813d095a749708f06f1faf65c0970", base_url="https://api.deepseek.com")

recipe_list_dict = {}
for data in train_data:
    dish_name = data['dish_name']
    recipe = data['recipe']
    if dish_name in recipe_list_dict:
        recipe_list_dict[dish_name].append(recipe)
    else:
        recipe_list_dict[dish_name] = [recipe]
dish_names = list(recipe_list_dict.keys())
dense_retriever = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
dish_embeddings = dense_retriever.encode(dish_names, device='cuda')

def get_prompt(index):
    with open(f'./log_base/ask_{index}.txt', 'r', encoding='utf-8') as f:
        temp = f.read()
        temp = temp.replace("```", '')
        temp = temp.replace("json", '').strip()
        if temp[0] != '[':
            temp = '[' + temp + ']'
        raw_pred_dish_name = eval(temp)[0]['dish_name']
    raw_pred_dish_name_embedding = dense_retriever.encode(raw_pred_dish_name, device='cuda')
    similarities = dense_retriever.similarity(raw_pred_dish_name_embedding, dish_embeddings)
    top_k = 500
    top_k_indices = torch.topk(similarities, top_k).indices.cpu().numpy()[0]
    exp_input = []
    exp_output = []
    target = [test_data[index]]
    for index in top_k_indices:
        dish_name = dish_names[index]
        exp_output.append({'dish_name': dish_name})
        exp_input.append({'recipe': recipe_list_dict[dish_name][0]})
    example = ""
    for i in range(len(exp_input)):
        example += f"输入：[{exp_input[i]}]\n输出：[{exp_output[i]}]\n"
    prompt = f'''现在你是一个厨师，你需要根据以下食谱制作一道菜品。请你根据食谱的描述，预测这道菜的名称。
输入的格式是一个json对象，包含以下字段：
- recipe: 一个字符串，表示食谱的描述。
输出的格式是一个json对象，包含以下字段：
- dish_name: 一个字符串，表示你预测的菜品名称。
以下是一些例子：
{example}接下来我会给你相应的输入（每次输入都只有一个菜谱，因此输出也只有一个），请你给出输出（只需要给出一个json对象，包含dish_name字段，不要有任何前后缀），这是输入：
{target}
那么输出是：'''
    return prompt

for index in tqdm(range(len(test_data))):
    prompt = get_prompt(index)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    temp = str(response.choices[0].message.content)
    with open(f'./log_retrieve/ask_{index}.txt', 'w', encoding='utf-8') as f:
        f.write(temp + '\n')
        
result = []
for i in range(1000):
    with open(f'./log_retrieve/ask_{i}.txt', 'r', encoding='utf-8') as f:
        temp = f.read()
        temp = temp.replace("```", '')
        temp = temp.replace("json", '').strip()
        if temp[0] != '[':
            temp = '[' + temp + ']'
        result.extend(eval(temp)[:1])
        
df = pd.DataFrame(result)
df['id'] = df.index
df = df[['id', 'dish_name']]
df.to_csv('./retrieve_result.csv', index=False)