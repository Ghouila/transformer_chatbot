import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Chatbot loop
print("Chatbot is ready! Type 'quit' to stop.")
step = 0
chat_history_ids = None

while True:
    # Take user input
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break
    
    # Encode the input and append it to the chat history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    if step > 0:
        # Append the new user input to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        attention_mask=attention_mask
    )

    # Decode the response and print it
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Chatbot: {bot_response}")
    
    step += 1
