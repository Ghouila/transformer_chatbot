import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# Load tokenizer (you can use GPT-2 or any other tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_len=40):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        src, trg = self.conversations[idx]
        
        # Tokenize input and output
        src_tokens = self.tokenizer(
            src, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt"
        )
        trg_tokens = self.tokenizer(
            trg, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt"
        )
        
        return src_tokens['input_ids'].squeeze(), trg_tokens['input_ids'].squeeze()

# Example conversations (replace with a real dataset)
conversations = [
    ("Hi, how are you?", "I'm good, thanks!"),
    ("What's your name?", "I'm a chatbot."),
    ("What do you do?", "I chat with people!")
]

# Create dataset and dataloader
dataset = ConversationDataset(conversations, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

import torch.nn as nn
from transformers import GPT2Model

class ChatbotTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=8, num_layers=6, max_len=40):
        super(ChatbotTransformer, self).__init__()
        
        self.gpt = GPT2Model.from_pretrained("gpt2")
        self.fc_out = nn.Linear(self.gpt.config.hidden_size, vocab_size)
    
    def forward(self, src_input_ids, trg_input_ids):
        src_outputs = self.gpt(input_ids=src_input_ids)
        decoder_outputs = self.gpt(input_ids=trg_input_ids, past_key_values=src_outputs.past_key_values)
        logits = self.fc_out(decoder_outputs.last_hidden_state)
        
        return logits
import torch.optim as optim

# Initialize model, loss, optimizer
model = ChatbotTransformer(vocab_size=len(tokenizer)).to("cuda")
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for src, trg in dataloader:
        src = src.to("cuda")
        trg = trg.to("cuda")
        
        optimizer.zero_grad()
        
        # Shift the target tokens by 1 for teacher forcing
        output = model(src, trg[:, :-1])
        
        # Reshape the output to compute the loss
        output = output.view(-1, output.size(-1))
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")
