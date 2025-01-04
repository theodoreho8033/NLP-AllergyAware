import torch
import json
import pandas as pd
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
import re
import logging
from sklearn.model_selection import train_test_split

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
MAX_LENGTH = 512
NUM_SAMPLES = 1
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALLERGEN_DB_PATH = "allergen_substitutes.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_recipes():
    print("Loading recipe dataset...")
    recipes = pd.read_csv('RAW_recipes.csv', encoding='utf8')
    recipes = recipes[['name', 'ingredients', 'steps']]
    recipes['name'] = recipes['name'].apply(lambda x: [x] if isinstance(x, str) else x)
    recipes['steps'] = recipes['steps'].apply(ast.literal_eval)
    recipes['ingredients'] = recipes['ingredients'].apply(ast.literal_eval)

    subsample_size = 100
    recipes = recipes.sample(n=subsample_size, random_state=42)
    logging.info(f"Using a subsample of {len(recipes)} recipes")

    def format_recipe(row):
        try:
            prompt = f"Name: {' '.join(row['name'])}\nIngredients: {' '.join(row['ingredients'])}\nInstructions:"
            completion = '\n'.join(row['steps'])
            return prompt + completion
        except Exception as e:
            logging.warning(f"Error processing recipe: {str(e)}")
            return None

    formatted_data = recipes.apply(format_recipe, axis=1)
    formatted_data = formatted_data[formatted_data.notna()].tolist()
    copy_formatted_data = formatted_data.copy()
    train_val_text, test_texts = train_test_split(copy_formatted_data, test_size=0.1, random_state=42)
    train_texts, val_texts = train_test_split(train_val_text, test_size=1/9, random_state=42)

    logging.info(f"Split dataset into {len(train_texts)} training, {len(val_texts)} validation samples and {len(test_texts)} test samples")
    return train_texts, val_texts, test_texts

def load_allergen_database():
    """Load and prepare the allergen substitution database for RAG"""
    with open(ALLERGEN_DB_PATH, 'r') as f:
        allergen_data = json.load(f)

    documents = []
    for allergen, info in allergen_data.items():
        doc = f"Allergen: {allergen}\nSubstitutes: {', '.join(info['substitutes'])}\nNotes: {info['notes']}"
        documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.create_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

def substitute_allergens(recipe_text, vectorstore):
    """Use RAG to identify and substitute allergens in the recipe"""
    ingredients_match = re.search(r"Ingredients:(.*?)Instructions:", recipe_text, re.DOTALL)
    if not ingredients_match:
        return recipe_text, []

    ingredients_text = ingredients_match.group(1).strip()
    ingredients_list = [i.strip() for i in ingredients_text.split('\n') if i.strip()]

    substitutions_made = []
    modified_recipe = recipe_text

    for ingredient in ingredients_list:
        results = vectorstore.similarity_search(
            f"What are substitutes for {ingredient}?",
            k=1
        )

        if results and "Substitutes:" in results[0].page_content:
            substitute_info = results[0].page_content.split("Substitutes:")[1].split("Notes:")[0].strip()
            substitutes = substitute_info.split(', ')[0]

            modified_recipe = modified_recipe.replace(ingredient, f"{substitutes} (substitute for {ingredient})")
            substitutions_made.append({
                "original": ingredient,
                "substitute": substitutes
            })

    return modified_recipe, substitutions_made

def generate_recipe(model, tokenizer, ingredients, name, vectorstore):
    prompt = f"""You are an expert chef and recipe writer. Given a recipe name and a list of ingredients, create a high-quality, detailed recipe.

Create a detailed recipe for: {name}
Using these ingredients: {ingredients}

Recipe:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device=model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.75,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=4,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    original_recipe = generated_recipe.replace(prompt, "").strip()

    print("\n=== Original Recipe ===")
    print(original_recipe)

    modified_recipe, substitutions = substitute_allergens(original_recipe, vectorstore)

    if substitutions:
        print("\n=== Allergen Substitutions Made ===")
        for sub in substitutions:
            print(f"Replaced '{sub['original']}' with '{sub['substitute']}'")

    print("\n=== Modified Recipe ===")
    print(modified_recipe)

    return {
        "original_recipe": original_recipe,
        "modified_recipe": modified_recipe,
        "substitutions": substitutions
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading allergen database and creating vector store...")
    vectorstore = load_allergen_database()
    train_text,val_text,test_text=load_recipes()
    recipe_data =test_text[:500]

    print(f"Generating {NUM_SAMPLES} recipes with allergen substitutions...")
    recipes = []
    for recipe in tqdm(recipe_data):
        parts = recipe.split("\nInstructions:")
        header = parts[0].split("\nIngredients: ")
        name = header[0].replace("Name: ", "")
        ingredients = header[1]
        reference_instructions = parts[1] if len(parts) > 1 else ""
        print("\n" + "="*50)
        print(f"Generating recipe for: {name}")
        print("="*50)
        generated_recipe = generate_recipe(model, tokenizer, ingredients,name,vectorstore)
        print("\n=== Complete Recipe Information ===")
        print("Original Recipe:", generation_result['original_recipe'])
        print("\nSubstitutions Made:", generation_result['substitutions'])
        print("\nModified Recipe:", generation_result['modified_recipe'])
        #recipe_dict = {
        #    "name": name,
        #    "ingredients": ingredients,
        #    "generation": generated_recipe,
        #    "reference_instructions": reference_instructions

        #}
        #recipes.append(recipe_dict)
        if len(recipes) % 50 == 0:
            temp_file = f"generated_recipes_temp_{len(recipes)}.json"
            with open(temp_file, "w") as f:
                json.dump(recipes, f, indent=4)
            print(f"\nProgress saved to {temp_file}")

    output_file = "generated_recipes_500.json"
    with open(output_file, "w") as f:
        json.dump(recipes, f, indent=4)

    print(f"\nAll recipes generated and saved to {output_file}")

if __name__ == "__main__":
    main()