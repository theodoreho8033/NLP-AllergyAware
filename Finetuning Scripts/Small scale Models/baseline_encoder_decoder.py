import os
import ast
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import logging
import random
import traceback
import numpy as np

from transformers import GPT2Tokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_key'


from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


dataset = 'shuyangli94/food-com-recipes-and-user-interactions'
dataset_dir = 'datasets/foodcom'

if not os.path.exists(os.path.join(dataset_dir, 'PP_recipes.csv')):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    api.dataset_download_files(dataset, path=dataset_dir, unzip=True)
    print('Dataset downloaded and extracted.')
else:
    print('Dataset already exists.')


recipes = pd.read_csv(os.path.join(dataset_dir, 'PP_recipes.csv'))


recipes = recipes[recipes['name_tokens'].apply(lambda x: len(ast.literal_eval(x)) > 0)]
recipes = recipes[recipes['ingredient_tokens'].apply(lambda x: len(ast.literal_eval(x)) > 0)]
recipes = recipes[recipes['steps_tokens'].apply(lambda x: len(ast.literal_eval(x)) > 0)]


recipes = pd.read_csv(os.path.join(dataset_dir, 'PP_recipes.csv'))
logger.info(f"Loaded {len(recipes)} recipes")


subsample_size = 500  # or use the full dataset if possible
recipes_subsample = recipes.sample(n=subsample_size, random_state=42)
logger.info(f"Using a subsample of {len(recipes_subsample)} recipes")


train_data, test_data = train_test_split(recipes_subsample, test_size=0.1, random_state=42)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})


class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.name_tokens = self.data['name_tokens'].values
        self.ingredient_tokens = self.data['ingredient_tokens'].values
        self.steps_tokens = self.data['steps_tokens'].values
        self.tokenizer = tokenizer
        logger.info(f"Dataset initialized with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def safe_eval(self, x):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            logger.warning(f"Failed to evaluate: {x}. Error: {str(e)}")
            return []

    def __getitem__(self, idx):
        try:
            name = self.safe_eval(self.name_tokens[idx])
            ingredients = self.safe_eval(self.ingredient_tokens[idx])
            steps = self.safe_eval(self.steps_tokens[idx])


            ingredients = [token for sublist in ingredients for token in sublist if isinstance(token, int)]


            sep_token_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
            input_tokens = name + [sep_token_id] + ingredients
            target_tokens = steps

            input_tokens = input_tokens[:512]
            target_tokens = target_tokens[:512]

            if not input_tokens or not target_tokens:
                logger.warning(f"Empty sequence found at index {idx}")
                return None

            return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error processing item at index {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            return None


train_dataset = RecipeDataset(train_data, tokenizer)
test_dataset = RecipeDataset(test_data, tokenizer)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        logger.warning("Empty batch encountered")
        return torch.tensor([]), torch.tensor([])
    
    inputs, targets = zip(*batch)
    
    logger.debug(f"Batch size: {len(inputs)}")
    logger.debug(f"Input sequence lengths: {[len(seq) for seq in inputs]}")
    logger.debug(f"Target sequence lengths: {[len(seq) for seq in targets]}")
    
    inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
    return inputs, targets


batch_size = 128 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.attention = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        src_len = src.size(1)
        trg_len = trg.size(1) if trg is not None else 100
        vocab_size = self.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

        embedded_src = self.embedding(src)
        encoder_outputs, hidden = self.encoder(embedded_src)
        
        # Combine bidirectional hidden states
        hidden = hidden.view(2, 2, batch_size, -1).transpose(0, 1).contiguous()
        hidden = hidden.view(batch_size, -1).unsqueeze(0)
        
        # Use only the last layer of the encoder as the initial hidden state for the decoder
        hidden = hidden[:, :, :self.decoder.hidden_size].contiguous()

        input = src[:, -1].unsqueeze(1)

        for t in range(trg_len):
            embedded_input = self.embedding(input)
            
            query = hidden.transpose(0, 1)
            keys = encoder_outputs
            
            energy = torch.tanh(self.attention(torch.cat((query.repeat(1, src_len, 1), keys), dim=2)))
            attention = self.v(energy).squeeze(2)
            attention_weights = torch.softmax(attention, dim=1)
            
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            
            rnn_input = torch.cat((embedded_input, context), dim=2)
            
            output, hidden = self.decoder(rnn_input, hidden)
            
            prediction = self.fc_out(output)
            
            outputs[:, t] = prediction.squeeze(1)

            if trg is not None:
                teacher_force = random.random() < teacher_forcing_ratio
                input = trg[:, t].unsqueeze(1) if teacher_force else prediction.argmax(2)
            else:
                input = prediction.argmax(2)

        return outputs


VOCAB_SIZE = len(tokenizer)
EMBED_DIM = 256
HIDDEN_DIM = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)


optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    valid_batches = 0
    total_batches = len(train_loader)
    
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(train_loader):
        logger.debug(f"Processing batch {batch_idx+1}/{total_batches}")
        
        src, trg = batch
        
        if src.numel() == 0 or trg.numel() == 0:
            logger.warning(f"Skipping empty batch {batch_idx+1}")
            continue
        
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        try:
            logger.debug(f"Running model forward pass")
            output = model(src, trg)
            logger.debug(f"Model output shape: {output.shape}")
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            logger.debug(f"Reshaped output: {output.shape}, Reshaped target: {trg.shape}")
            
            loss = criterion(output, trg)
            logger.debug(f"Calculated loss: {loss.item()}")
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}")
        except Exception as e:
            logger.error(f"Error in batch {batch_idx+1}: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
    
    if valid_batches > 0:
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss/valid_batches:.4f}')
    else:
        logger.warning(f'Epoch {epoch+1}/{num_epochs}, No valid batches')


logger.info("Training completed")

# After training, save the model
torch.save(model.state_dict(), 'seq2seq_model.pth')
print('Model saved to seq2seq_model.pth')

# Later, when you want to use the model for inference:
try:
    model.load_state_dict(torch.load('seq2seq_model.pth'))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please ensure 'seq2seq_model.pth' exists.")
except Exception as e:
    print(f"An error occurred while loading the model: {str(e)}")

def generate_recipe(model, tokenizer, input_text, max_length=100):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model(input_ids, trg=None)
        generated_ids = output.argmax(dim=-1)
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

input_text = "Generate a recipe for chocolate cake:"
generated_recipe = generate_recipe(model, tokenizer, input_text)
print("Generated Recipe:")
print(generated_recipe)
