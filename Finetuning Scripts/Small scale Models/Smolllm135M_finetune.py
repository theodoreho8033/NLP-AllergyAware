import os
import torch
import logging
import json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

torch.backends.cudnn.enabled = False

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
DATASET_NAME = "shuyangli94/food-com-recipes-and-user-interactions"
OUTPUT_DIR = "recipe_model_output"
MAX_LENGTH = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
TEST_SAMPLES = 500


os.environ['KAGGLE_USERNAME'] = 'username'
os.environ['KAGGLE_KEY'] = 'keyhere'

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


dataset = 'shuyangli94/food-com-recipes-and-user-interactions'
dataset_dir = 'datasets/foodcom'

if not os.path.exists(os.path.join(dataset_dir, 'RAW_recipes.csv')):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    api.dataset_download_files(dataset, path=dataset_dir, unzip=True)
    print('Dataset downloaded and extracted.')
else:
    print('Dataset already exists.')


def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/training_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    config = {
        'model_name': MODEL_NAME,
        'dataset_name': DATASET_NAME,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS
    }
    logging.info(f"Training configuration: {json.dumps(config, indent=2)}")

    return log_filename

def load_and_preprocess_data():
    logging.info("Loading and preprocessing dataset...")

    try:
       from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
       logging.error("Kaggle API not found. Please install it using: pip install kaggle")
       raise
    os.environ['KAGGLE_USERNAME'] = 'username'
    os.environ['KAGGLE_KEY'] = 'key'

    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(kaggle_dir):
       os.makedirs(kaggle_dir)

    kaggle_config = {
       "username": "username",
       "key": "key"
    }
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(kaggle_json_path, 'w') as f:
       json.dump(kaggle_config, f)


    os.chmod(kaggle_json_path, 0o600)

    api = KaggleApi()
    api.authenticate()

    dataset_dir = 'datasets/foodcom'
    if not os.path.exists(os.path.join(dataset_dir, 'RAW_recipes.csv')):
       if not os.path.exists(dataset_dir):
           os.makedirs(dataset_dir)
       api.dataset_download_files(DATASET_NAME, path=dataset_dir, unzip=True)
       logging.info('Dataset downloaded and extracted.')
    else:
       logging.info('Dataset already exists.')



    recipes = pd.read_csv('/home/anneketh/datasets/foodcom/RAW_recipes.csv')
    recipes=recipes[['name','ingredients','steps']]
    recipes['name'] = recipes['name'].apply(lambda x: [x] if isinstance(x, str) else x)
    recipes['steps']=recipes['steps'].apply(ast.literal_eval)
    recipes['ingredients']=recipes['ingredients'].apply(ast.literal_eval)
    subsample_size = 100000
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
    copy_formatted_data=formatted_data.copy()
    train_val_text,test_texts=train_test_split(copy_formatted_data,test_size=0.1,random_state=42)
    train_texts,val_texts=train_test_split(train_val_text,test_size=1/9,random_state=42)


    logging.info(f"Split dataset into {len(train_texts)} training ,{len(val_texts)} validation samples and {len(test_texts)} test samples")
    return train_texts, val_texts,test_texts

def tokenize_data(texts, tokenizer):
    logging.info(f"Tokenizing {len(texts)} texts...")
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors='pt'
    )

class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def test_model(model, tokenizer, ingredients,name):
    prompt = f"""Given recipe name {name} and these ingredients: {ingredients}
Write a detailed recipe with:
1. List of all ingredients with measurements
2. Step-by-step cooking instructions
3. Cooking time and temperature if needed
4. Serving suggestions

Recipe:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_return_sequences=1,
        temperature=0.75,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=4,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info("Generated recipe:")
    logging.info(generated_recipe)

    return generated_recipe

def calculate_metrics(reference, hypothesis, ingredients=None):
    """Enhanced metric calculation with additional recipe-specific metrics"""
    metrics = {}
    
    # 1. Standard ROUGE and BLEU scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    try:
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Basic ROUGE and BLEU metrics
        metrics.update({
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'rouge1_precision': rouge_scores['rouge1'].precision,
            'rouge1_recall': rouge_scores['rouge1'].recall,
            'rouge2_precision': rouge_scores['rouge2'].precision,
            'rouge2_recall': rouge_scores['rouge2'].recall,
            'rougeL_precision': rouge_scores['rougeL'].precision,
            'rougeL_recall': rouge_scores['rougeL'].recall,
        })
        
        # BLEU scores
        metrics.update({
            'bleu1': sentence_bleu([reference_tokens], hypothesis_tokens,
                                weights=(1, 0, 0, 0),
                                smoothing_function=SmoothingFunction().method1),
            'bleu2': sentence_bleu([reference_tokens], hypothesis_tokens,
                                weights=(0.5, 0.5, 0, 0),
                                smoothing_function=SmoothingFunction().method1),
            'bleu3': sentence_bleu([reference_tokens], hypothesis_tokens,
                                weights=(0.33, 0.33, 0.33, 0),
                                smoothing_function=SmoothingFunction().method1),
            'bleu4': sentence_bleu([reference_tokens], hypothesis_tokens,
                                smoothing_function=SmoothingFunction().method1),
        })
        
        # 2. Perplexity score
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        inputs = tokenizer(hypothesis, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        metrics['perplexity'] = perplexity
        
        # 3. Ingredient Coverage Score
        if ingredients:
            ingredient_list = [ing.strip().lower() for ing in ingredients.split(',')]
            mentioned_ingredients = sum(1 for ing in ingredient_list if ing in hypothesis.lower())
            metrics['ingredient_coverage'] = mentioned_ingredients / len(ingredient_list)
        else:
            metrics['ingredient_coverage'] = 0.0
        
        # 4. Step Complexity Score
        steps = [s for s in hypothesis.split('\n') if s.strip()]
        avg_step_length = np.mean([len(nltk.word_tokenize(step)) for step in steps]) if steps else 0
        metrics['step_complexity'] = min(1.0, avg_step_length / 20)  # Normalize to [0,1]
        
        # 5. Recipe Coherence Score (using step order logic)
        cooking_verbs = set(['mix', 'stir', 'cook', 'bake', 'boil', 'fry', 'add', 'combine', 'heat', 'pour'])
        verb_sequence = []
        for step in steps:
            tokens = nltk.word_tokenize(step.lower())
            verbs = [token for token in tokens if token in cooking_verbs]
            if verbs:
                verb_sequence.append(verbs[0])
        
        # Check if verbs follow logical cooking order
        valid_transitions = {
            'mix': ['add', 'pour', 'stir'],
            'add': ['mix', 'stir', 'combine'],
            'combine': ['mix', 'stir', 'heat'],
            'heat': ['cook', 'bake', 'boil', 'fry'],
            'stir': ['add', 'pour', 'heat']
        }
        
        coherence_score = 0
        if len(verb_sequence) > 1:
            valid_transitions_count = sum(1 for i in range(len(verb_sequence)-1) 
                                     if verb_sequence[i+1] in valid_transitions.get(verb_sequence[i], []))
            coherence_score = valid_transitions_count / (len(verb_sequence) - 1)
        metrics['recipe_coherence'] = coherence_score
        
        # 6. Temperature and Time Specification Score
        temp_patterns = r'\d+\s*[°℉℃F]'
        time_patterns = r'\d+\s*(minutes?|mins?|hours?|hrs?)'
        
        has_temp = bool(re.search(temp_patterns, hypothesis))
        has_time = bool(re.search(time_patterns, hypothesis))
        metrics['temp_time_score'] = (has_temp + has_time) / 2
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        default_metrics = {
            'perplexity': float('inf'),
            'ingredient_coverage': 0.0,
            'step_complexity': 0.0,
            'recipe_coherence': 0.0,
            'temp_time_score': 0.0
        }
        metrics.update(default_metrics)
    
    return metrics

def visualize_recipe_metrics(metrics_df, output_path="recipe_metrics_radar.png"):
    """Create a radar chart for recipe-specific metrics"""
    metrics_to_plot = [
        'ingredient_coverage', 
        'step_complexity', 
        'recipe_coherence',
        'temp_time_score',
        'rouge1',
        'bleu1'
    ]
    

    num_vars = len(metrics_to_plot)
    

    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    

    values = metrics_df[metrics_to_plot].mean().values.flatten().tolist()
    values += values[:1]
    

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    
    plt.title("Recipe Generation Quality Metrics")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, tokenizer, test_data, model_name="unnamed"):

    model.eval()
    metrics = []

    num_samples = min(TEST_SAMPLES,len(test_data))
    logging.info(f"Evaluating {model_name} on {num_samples} samples...")

    for recipe in tqdm(test_data[:num_samples]):
        parts = recipe.split("\nInstructions:")
        header = parts[0].split("\nIngredients: ")
        name = header[0].replace("Name: ", "")
        ingredients = header[1]
        reference_instructions = parts[1] if len(parts) > 1 else ""

        generated_recipe = test_model(model, tokenizer, ingredients, name)

        scores = calculate_metrics(reference_instructions, generated_recipe, ingredients)
        metrics.append(scores)


    avg_metrics = {
        metric: sum(m[metric] for m in metrics) / len(metrics)
        for metric in metrics[0].keys()
    }

    logging.info(f"{model_name} Metrics: {json.dumps(avg_metrics, indent=2)}")
    return avg_metrics

def generate_evaluation_report(baseline_metrics, finetuned_metrics, test_cases, model_outputs, training_time):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f'evaluation_reports/{timestamp}'
    os.makedirs(report_dir, exist_ok=True)

    with open(f'{report_dir}/evaluation_report.md', 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Configuration\n\n")
        f.write(f"- Base Model: {MODEL_NAME}\n")
        f.write(f"- Training Epochs: {NUM_EPOCHS}\n")
        f.write(f"- Batch Size: {BATCH_SIZE}\n")
        f.write(f"- Learning Rate: {LEARNING_RATE}\n")
        f.write(f"- Training Time: {training_time:.2f} seconds\n\n")

        f.write("## Performance Metrics\n\n")
        metrics_table = []
        headers = ["Metric", "Baseline", "Fine-tuned", "Improvement", "% Change"]

        for metric in baseline_metrics.keys():
            baseline = baseline_metrics[metric]
            finetuned = finetuned_metrics[metric]
            improvement = finetuned - baseline
            pct_change = (improvement / baseline) * 100 if baseline != 0 else float('inf')

            metrics_table.append([
                metric,
                f"{baseline:.4f}",
                f"{finetuned:.4f}",
                f"{improvement:+.4f}",
                f"{pct_change:+.2f}%"
            ])

        f.write(tabulate(metrics_table, headers=headers, tablefmt="pipe"))
        f.write("\n\n")

        plot_metrics_comparison(baseline_metrics, finetuned_metrics, report_dir)
        f.write(f"![Metrics Comparison]({report_dir}/metrics_comparison.png)\n\n")

        f.write("## Sample Generations Analysis\n\n")
        for idx, test_case in enumerate(test_cases):
            f.write(f"### Test Case {idx + 1}\n\n")
            f.write(f"**Recipe Name:** {test_case['name']}\n\n")
            f.write(f"**Ingredients:**\n{test_case['ingredients']}\n\n")
            f.write(f"**Baseline Generation:**\n{model_outputs['baseline'][idx]}\n\n")
            f.write(f"**Fine-tuned Generation:**\n{model_outputs['finetuned'][idx]}\n\n")

            case_metrics = calculate_metrics(
                model_outputs['baseline'][idx],
                model_outputs['finetuned'][idx]
            )
            f.write("**Generation Metrics:**\n")
            for metric, value in case_metrics.items():
                f.write(f"- {metric}: {value:.4f}\n")
            f.write("\n")

        f.write("## Additional Analysis\n\n")

        baseline_lengths = [len(x.split()) for x in model_outputs['baseline']]
        finetuned_lengths = [len(x.split()) for x in model_outputs['finetuned']]

        f.write("### Generation Length Statistics\n\n")
        length_stats = {
            "Model": ["Baseline", "Fine-tuned"],
            "Avg Length": [np.mean(baseline_lengths), np.mean(finetuned_lengths)],
            "Min Length": [np.min(baseline_lengths), np.min(finetuned_lengths)],
            "Max Length": [np.max(baseline_lengths), np.max(finetuned_lengths)]
        }
        f.write(tabulate(pd.DataFrame(length_stats), headers="keys", tablefmt="pipe"))
        f.write("\n\n")

        f.write("### Vocabulary Usage\n\n")
        baseline_vocab = set(' '.join(model_outputs['baseline']).split())
        finetuned_vocab = set(' '.join(model_outputs['finetuned']).split())
        common_words = len(baseline_vocab.intersection(finetuned_vocab))

        vocab_stats = {
            "Model": ["Baseline", "Fine-tuned"],
            "Unique Words": [len(baseline_vocab), len(finetuned_vocab)],
            "Common Words": [common_words, common_words],  # Duplicate value for both rows
            "Model-Specific Words": [
                len(baseline_vocab - finetuned_vocab),
                len(finetuned_vocab - baseline_vocab)
            ]
        }

        vocab_df = pd.DataFrame(vocab_stats)
        f.write(tabulate(vocab_df, headers="keys", tablefmt="pipe"))
        f.write("\n\n")

        # Add new sections for additional metrics
        f.write("\n## Recipe-Specific Metrics Analysis\n\n")
        
        f.write("### Ingredient Coverage\n")
        f.write("Measures how well the generated recipe utilizes the provided ingredients.\n\n")
        
        f.write("### Step Complexity\n")
        f.write("Analyzes the complexity and completeness of cooking instructions.\n\n")
        
        f.write("### Recipe Coherence\n")
        f.write("Evaluates logical flow and ordering of cooking steps.\n\n")
        
        f.write("### Temperature and Time Specifications\n")
        f.write("Tracks inclusion of crucial cooking parameters.\n\n")
        
        visualize_recipe_metrics(pd.DataFrame(finetuned_metrics), f'{report_dir}/recipe_metrics_radar.png')
        f.write(f"![Recipe Metrics Radar]({report_dir}/recipe_metrics_radar.png)\n\n")

    return f'{report_dir}/evaluation_report.md'

def plot_metrics_comparison(baseline_metrics, finetuned_metrics, report_dir):
    """Generate comparative visualizations of metrics."""
    plt.figure(figsize=(12, 6))

    metrics = list(baseline_metrics.keys())
    baseline_values = [baseline_metrics[m] for m in metrics]
    finetuned_values = [finetuned_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, baseline_values, width, label='Baseline')
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Baseline vs Fine-tuned Model Performance')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{report_dir}/metrics_comparison.png')
    plt.close()

def main():
    log_filename = setup_logging()
    logging.info("Starting training pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Using device: {device}")

    logging.info(f"Loading model and tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.float32).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        logging.info("Added padding token to tokenizer")

    train_texts, val_texts,test_texts = load_and_preprocess_data()

    logging.info(f"Using {len(test_texts)} samples for testing")

    start_time = time.time()

    logging.info("Loading baseline model for comparison...")
    baseline_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    baseline_model.to(device)
    baseline_metrics = evaluate_model(baseline_model, tokenizer, test_texts, "Baseline Model")

    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)

    train_dataset = RecipeDataset(train_encodings)
    val_dataset = RecipeDataset(val_encodings)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_steps=100,
        save_steps=100,
        learning_rate=LEARNING_RATE,
        fp16=False,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
    )

    class RecipeDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, examples):
            batch = super().__call__(examples)
            # Add position IDs
            batch["position_ids"] = torch.arange(batch["input_ids"].shape[1])[None, :]
            return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=RecipeDataCollator(tokenizer=tokenizer, mlm=False),
    )

    logging.info("Starting training...")
    trainer.train()

    training_time = time.time() - start_time

    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"Saved final model to {final_model_path}")

    finetuned_metrics = evaluate_model(model, tokenizer, test_texts, "Fine-tuned Model")

    test_cases = [
        {
            'name': 'Lemon Garlic Chicken',
            'ingredients': "chicken breast, garlic, olive oil, lemon, thyme"
        },
        {
            'name': 'Classic Vanilla Cake',
            'ingredients': "flour, sugar, eggs, butter, vanilla extract"
        },
        {
            'name': 'Vegetable Fried Rice',
            'ingredients': "rice, vegetables, soy sauce, ginger, sesame oil"
        }
    ]

    model_outputs = {
        'baseline': [],
        'finetuned': []
    }

    for case in test_cases:
        model_outputs['baseline'].append(
            test_model(baseline_model, tokenizer, case['name'], case['ingredients'])
        )
        model_outputs['finetuned'].append(
            test_model(model, tokenizer, case['name'], case['ingredients'])
        )

    report_path = generate_evaluation_report(
        baseline_metrics,
        finetuned_metrics,
        test_cases,
        model_outputs,
        training_time
    )

    logging.info(f"Evaluation report generated at: {report_path}")
    logging.info(f"Training and testing completed. Check {log_filename} for full logs")

if __name__ == "__main__":
    # Download all required NLTK data
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('tokenizers/punkt/english.pickle')
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {e}")
        raise

    main()
