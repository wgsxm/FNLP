import json
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import os

test_data = json.load(open('./dish-name-recognition/test_recipe.json', 'r', encoding='utf-8'))
os.makedirs('./log_base', exist_ok=True)
client = OpenAI(api_key="sk-7d2813d095a749708f06f1faf65c0970", base_url="https://api.deepseek.com")

prompt = '''
现在你是一个厨师，你需要根据以下食谱制作一道菜品。请你根据食谱的描述，预测这道菜的名称。
输入的格式是一个json对象，包含以下字段：
- recipe: 一个字符串，表示食谱的描述。
例如：
[
    {'recipe': '蛋清蛋黄分离，蛋清放在无水无油的容器里，加几滴柠檬汁放一边备用\n蛋黄加20克砂糖拌至糖完全融化\n加入酸奶拌匀\n加入色拉油拌匀\n在这个过程我加了那5克白兰地\n筛入低粉和澄粉\n用橡皮刮刀切拌均匀，切记不要划圈儿，以免面糊起筋\n将那40克白砂糖分三次加入蛋白糊中，用电动打蛋器将蛋白打至硬性发泡，提起打蛋器带有直立小尖角\n蛋白糊分三次加入蛋黄糊中，每一次都切拌均匀\n将混合好的面糊倒入八寸活底蛋糕模中，轻震几下，震出里面的大气泡\n烤箱150度预热，放入模具，烤箱倒数第二层，140度拷50分钟，转150度烤10分钟出炉\n将蛋糕倒扣在考网上，放凉后脱模即可'},
    {'recipe': '肥牛卷冷水下锅焯一下，去掉血沫，备用\n锅内放入少许食用油，放入圆葱中火炒至轻微变软\n放入焯好的肥牛卷，再加入适量的清水，加入适量的料酒、耗油、生抽，几滴老抽，少许精盐和白糖，中小火慢炖10分钟左右，加入适量水淀粉，汤汁粘稠即可关火\n碗内盛好米饭，铺好肥牛卷，浇上少许汤汁，撒少许芝麻，完成'}
]
输出的格式是一个json对象，包含以下字段：
- dish_name: 一个字符串，表示你预测的菜品名称。
例如：
[
    {'dish_name': '酸奶戚风蛋糕'},
    {'dish_name': '肥牛饭'}
]
接下来我会给你相应的输入，请你给出预测。
'''

for index in tqdm(range(len(test_data))):
    content = test_data[index:index+1]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
            {"role": "user", "content": str(content)}
        ],
        stream=False
    )
    temp = str(response.choices[0].message.content)
    with open(f'./log_base/ask_{index}.txt', 'w', encoding='utf-8') as f:
        f.write(temp + '\n')

result = []
for i in range(1000):
    with open(f'./log_base/ask_{i}.txt', 'r', encoding='utf-8') as f:
        temp = f.read()
        temp = temp.replace("```", '')
        temp = temp.replace("json", '')
        c = eval(temp)
        result.extend(eval(temp)[:1])
        
df = pd.DataFrame(result)
df['id'] = df.index
df = df[['id', 'dish_name']]
df.to_csv('./base_result.csv', index=False, encoding='utf-8')