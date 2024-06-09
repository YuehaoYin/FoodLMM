import json
import torch
import numpy as np
original_ingredients_path = 'dataset/Recipe1M/original_ingredients.json'
with open(original_ingredients_path) as f:
    original_ingredients = json.load(f)


original_ingredients_embeddings = dict()





def mapping_ingre(ingredient,mapping_data,ingredients_1488):
    for idx in range(len(mapping_data)):
        if ingredient==mapping_data[idx]['ingredient']:
            return ingredients_1488[mapping_data[idx]['idx']]

def find_similar_ingredient(ingredient, ingredients_embedding, tokenizer, model):
    max_sim = -1.0
    similar_ingredient = None
    ingredient_embedding = get_bert_embedding(tokenizer, model, ingredient)
    for ingredient, candidate_embedding in ingredients_embedding.items():
        sim = ingredient_embedding.dot(candidate_embedding) / (np.linalg.norm(ingredient_embedding) * np.linalg.norm(candidate_embedding))
        # sim= F.cosine_similarity(ingredient_embedding, candidate_embedding)
        if sim > max_sim:
            max_sim = sim
            similar_ingredient = ingredient

    return similar_ingredient




def get_bert_embedding(tokenizer, model, input_text):
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    last_hidden_states = model(input_ids.cuda())[0].cpu().detach().numpy()
    last_hidden_states = last_hidden_states[:, 0, :]
    last_hidden_states = last_hidden_states.flatten()
    return last_hidden_states


def ingredients_embedding(ingredients_list, tokenizer, model):
    ingredients_embeddings=dict()
    for ingredient in ingredients_list:
        ingredient_embedding = get_bert_embedding(tokenizer, model, ingredient)
        ingredients_embeddings[ingredient] = ingredient_embedding
    return ingredients_embeddings

