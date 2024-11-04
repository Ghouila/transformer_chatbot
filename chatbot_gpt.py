import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the pad token as eos token
tokenizer.pad_token = tokenizer.eos_token

# Function to generate chatbot responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Create attention mask
    attention_mask = inputs['attention_mask']
    
    # Generate the response with a maximum length of 100 tokens
    outputs = model.generate(
        inputs['input_ids'],  # Input IDs need to be explicitly passed here
        max_length=100, 
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,  # Enable sampling for diversity
        attention_mask=attention_mask,  # Pass attention mask
        pad_token_id=tokenizer.eos_token_id  # Ensure padding uses eos token
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Loop to interact with the user
print("Chatbot is ready! Type 'quit' to stop.")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break
    
    # Generate a chatbot response
    response = generate_response(user_input)
    print(f"Chatbot: {response}")


def encode_sentence(sentence, vocab, max_len):
            return [vocab['<sos>']] + [vocab[token] for token in sentence if token in vocab] + [vocab['<eos>']] + \
                [vocab['<pad>']] * (max_len - len(sentence) - 2)