from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

# Environment Setup
os.environ['CUDA_VISIBLE_DEVICES']= "0,1,2,3,4,5"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Embedding version
embedding_version = "distilbert-base-uncased"
# embedding_version = "bert-base-uncased"

# Output directory
output_dir = 'civil_comments_embeddings_distilbert'
# output_dir = 'amazon_embeddings_distilbert'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(embedding_version)
model = AutoModel.from_pretrained(embedding_version)
model.eval()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel
model.to(device)

# Load data
amazon_df = pd.read_csv('/lfs/mercury1/0/shinyw/wilds/scripts/verified_amazon_reviews.csv')
civil_df = pd.read_csv('/lfs/mercury1/0/shinyw/wilds/scripts/civil_df_cleaned.csv')

# Custom Dataset class for the text
class TextDataset(Dataset):
    def __init__(self, text_input, tokenizer, max_length=512):
        self.text_input = text_input
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_input)

    def __getitem__(self, idx):
        review = self.text_input[idx]
        encoding = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {key: value.squeeze(0) for key, value in encoding.items()}

# Function to compute embeddings for a batch of reviews using DataLoader
def compute_embeddings_dataloader_full(text_input, output_dir, batch_size=512, chunk_size=100352):
    dataset = TextDataset(text_input, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    chunk_index = 0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through batches using DataLoader
    for batch in tqdm(dataloader, desc="Processing batches", leave=False):  # Add progress bar for batches
        # Move input tensors to the correct device
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            # Get model outputs
            outputs = model(**batch)  # DataParallel handles splitting across GPUs
            # Extract the CLS embeddings for each review (first token [CLS])
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)

            # If chunk reaches the desired size, save and reset
            if len(embeddings) >= chunk_size:
                chunk_filename = os.path.join(output_dir, f'embeddings_chunk_{chunk_index}.npy')
                np.save(chunk_filename, np.array(embeddings))
                print(f"Saved chunk {chunk_index} to {chunk_filename}")

                # Reset for next chunk
                embeddings = []
                chunk_index += 1

    # Save any remaining embeddings if they're smaller than the chunk size
    if embeddings:
        chunk_filename = os.path.join(output_dir, f'embeddings_chunk_{chunk_index}.npy')
        np.save(chunk_filename, np.array(embeddings))
        print(f"Saved final chunk {chunk_index} to {chunk_filename}")


# Convert all text to embeddings - Amazon
# amazon_df['reviewText'] = amazon_df['reviewText'].astype(str)  # Ensure reviews are strings
# text_input = amazon_df['reviewText'].tolist()  # Extract all reviews as a list

# Convert all text to embeddings - CivilComments
civil_df['comment_text'] = civil_df['comment_text'].astype(str)
text_input = civil_df['comment_text'].tolist()

# Compute embeddings for all reviews in the dataset
compute_embeddings_dataloader_full(text_input, output_dir)
