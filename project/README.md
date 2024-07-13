# Dependency
- openai
- tqdm
- pandas
- sentence_transformers
- torch

# Code Structure
## base
- dish_base.py: a basic solution for dish recognition
1. read the data, create a dictionary for log, and create openai client
2. the prompt
3. a for-loop to generate the response
4. adress the response and save the response to a csv file
## retrieve
- dish_retrieve.py: a solution for dish recognition with retrieval(it is based on the base solution)
1. read the data, create a dictionary for log, and create openai client
2. embedding all **dish names** in train data
3. `get_prompt` function: generate the prompt. The function first read the predicted dish name in the base solution and then use the predicted dish name to retrieve the top 500 similar dish names in the train data. Finally, the function will return the prompt based on the top 500 similar dish names.
4. a for-loop to generate the response
5. adress the response and save the response to a csv file
