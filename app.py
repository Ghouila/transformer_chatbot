import torch
import torch.nn as nn
import torch.optim as optim
from Transformer import Transformer

def load_model(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device, max_len=100):
    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256,
        num_layers=6, forward_expansion=4, heads=8, dropout=0.1, device=device, max_length=max_len
    ).to(device)

    model.load_state_dict(torch.load("transformer_model_exemple_2.pth"))
    model.eval()  # Mettre le modèle en mode évaluation
    return model

if __name__ == "__main__":

    #######################
    # Exemple 1

    def exemple_1():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
        trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

        src_pad_idx = 0
        trg_pad_idx = 0
        src_vocab_size = 10
        trg_vocab_size = 10

        model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

        out = model(x, trg[:, :-1])
        print(out.shape)

    #####################
    # Exemple 2

    def exemple_2():
        """
        import torch.optim as optim
        import spacy
        from torchtext.data.utils import get_tokenizer
        from collections import Counter
        from torchtext.vocab import Vocab
        from torchtext.data import Field, BucketIterator
        from torchtext.datasets import Multi30k
        """
        import torch
        import torch.nn as nn
        from datasets import load_dataset
        from transformers import BertTokenizer, BertModel
        import torch.optim as optim

        # Load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Load HuggingFace dataset
        dataset = load_dataset("wmt14", "fr-en", split="train[:5%]")

        # Extract source (English) and target (French) sentences
        src_sentences = [ex['en'] for ex in dataset['translation'][:5]]
        trg_sentences = [ex['fr'] for ex in dataset['translation'][:5]]

        print(f"Source Sentences: {src_sentences[:5]}")
        print(f"Target Sentences: {trg_sentences[:5]}")

        # Define a function to encode sentences using BERT
        def encode_sentence_bert(sentence):
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=100)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            # Use the [CLS] token's embedding as the sentence embedding
            return outputs.last_hidden_state[:, 0, :]

        # Encode the source sentences with BERT
        src_embeddings = [encode_sentence_bert(sentence) for sentence in src_sentences]
        trg_embeddings = [encode_sentence_bert(sentence) for sentence in trg_sentences]

        # Stack embeddings to create the tensors
        src_tensor = torch.stack(src_embeddings).squeeze(1)  # Shape: [batch_size, embed_size]
        trg_tensor = torch.stack(trg_embeddings).squeeze(1)  # Shape: [batch_size, embed_size]

        # Initialize the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Transformer(
            src_vocab_size=768,  # Using BERT's embedding size of 768
            trg_vocab_size=768,  # Same for target
            src_pad_idx=0,       # Dummy value, not needed for BERT embeddings
            trg_pad_idx=0,       # Dummy value, not needed for BERT embeddings
            embed_size=768,      # Match BERT's embedding size
            num_layers=6, forward_expansion=4, heads=8, dropout=0.1, device=device, max_length=100
        ).to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        # Define the training function
        def train_step(src, trg):
            src = src.to(device)  # Move the input to the device
            trg = trg.to(device)  # Move the target to the device
            model.train()
            optimizer.zero_grad()

            trg_input = trg[:, :-1]
            outputs = model(src, trg_input)

            # Reshape the outputs and trg for loss computation
            outputs = outputs.reshape(-1, outputs.shape[2])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg)
            loss.backward()
            optimizer.step()

            return loss.item()

        # Training the model with the BERT embeddings
        N_EPOCHS = 10
        for epoch in range(N_EPOCHS):
            loss = train_step(src_tensor, trg_tensor)
            if epoch % 1 == 0:
                print(f"Epoch {epoch} Loss: {loss:.4f}")

        # Save the model after training
        torch.save(model.state_dict(), "transformer_model_bert.pth")

        # Exemple de prédiction
        def translate_sentence(model, sentence, src_vocab, trg_vocab, max_len, device="cuda"):
            # Tokenizer source sentence
            tokens = [src_vocab.stoi["<sos>"]] + [src_vocab.stoi[word] for word in sentence.split()] + [src_vocab.stoi["<eos>"]]
            if len(tokens) > max_len:
                tokens = tokens[:max_len-1] + [src_vocab.stoi["<eos>"]]

            # Convert to tensor and add batch dimension
            src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

            # Create source mask
            src_mask = model.make_src_mask(src_tensor)

            # Encode the source sentence
            with torch.no_grad():
                enc_src = model.encoder(src_tensor, src_mask)

            # Initialize target sentence with <sos> token
            trg_tokens = [trg_vocab.stoi["<sos>"]]

            for _ in range(max_len):
                trg_tensor = torch.LongTensor(trg_tokens).unsqueeze(0).to(device)
                trg_mask = model.make_trg_mask(trg_tensor)

                # Get prediction
                with torch.no_grad():
                    output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)

                # Get the token with the highest probability
                best_guess = output.argmax(2)[:, -1].item()
                trg_tokens.append(best_guess)

                # If we predict <eos> we stop translating
                if best_guess == trg_vocab.stoi["<eos>"]:
                    break
            translated_sentence = [trg_vocab.itos[idx] for idx in trg_tokens]

            # Remove the starting <sos> token
            return translated_sentence[1:]

        # Example translation
        translated_sentence = translate_sentence(model, "Transformers are not that complicated", tokenizer, max_len=100)
        print(f"Translated Sentence: {' '.join(translated_sentence)}")

        pass

    ######################"
    # Exemple 3"

    def exemple_3():
        from torchtext.data.utils import get_tokenizer
        from collections import Counter
        from torchtext.vocab import Vocab

        # Définition du modèle Transformer (pas inclus ici, j'imagine que tu l'as déjà)
        # from transformer_model import Transformer

        trained = 0

        # Tokenisation et vocabulaire
        tokenizer = get_tokenizer("basic_english")

        # Exemples de phrases sources et cibles
        src_sentences = ["This is a test sentence", "Transformers are powerful models", "Learning PyTorch is fun"]
        trg_sentences = ["Ceci est une phrase de test", "Les transformers sont des modèles puissants", "Apprendre PyTorch est amusant"]

        # Tokenisation des phrases sources et cibles
        src_tokenized = [tokenizer(sentence) for sentence in src_sentences]
        trg_tokenized = [tokenizer(sentence) for sentence in trg_sentences]

        # Construction du vocabulaire
        src_counter = Counter([token for sentence in src_tokenized for token in sentence])
        trg_counter = Counter([token for sentence in trg_tokenized for token in sentence])

        # Création des vocabulaires
        src_vocab = Vocab(src_counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
        trg_vocab = Vocab(trg_counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'])

        # Fonction pour convertir les phrases en indices
        def encode_sentence(sentence, vocab, max_len):
            return [vocab['<sos>']] + [vocab[token] for token in sentence] + [vocab['<eos>']] + [vocab['<pad>']] * (max_len - len(sentence) - 2)

        # Définition de la longueur maximale des séquences
        max_len = 10

        # Conversion des phrases en indices
        src_encoded = [encode_sentence(sentence, src_vocab, max_len) for sentence in src_tokenized]
        trg_encoded = [encode_sentence(sentence, trg_vocab, max_len) for sentence in trg_tokenized]

        # Définir le périphérique (GPU ou CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Conversion en tenseurs PyTorch sur le bon périphérique
        src_tensor = torch.tensor(src_encoded).to(device)
        trg_tensor = torch.tensor(trg_encoded).to(device)

        # Paramètres du modèle
        src_vocab_size = len(src_vocab)
        trg_vocab_size = len(trg_vocab)
        src_pad_idx = src_vocab['<pad>']
        trg_pad_idx = trg_vocab['<pad>']

        # Initialisation du modèle
        model = Transformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256,
            num_layers=6, forward_expansion=4, heads=8, dropout=0.1, device=device, max_length=max_len
        ).to(device)

        # Fonction d'entraînement
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        # Fonction d'entraînement
        def train_step(src, trg):
            model.train()
            optimizer.zero_grad()

            trg_input = trg[:, :-1]  # Décalage pour entraîner le modèle sur la séquence cible
            outputs = model(src, trg_input)

            # Réorganiser la sortie pour la fonction de perte
            outputs = outputs.reshape(-1, outputs.shape[2])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg)
            loss.backward()
            optimizer.step()

            return loss.item()

        # Entraînement du modèle avec des phrases réelles
        if not trained:
            for epoch in range(2000):
                loss = train_step(src_tensor, trg_tensor)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch} Loss: {loss:.4f}")
        else:
            model = load_model(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device)

        # Sauvegarder les poids du modèle après l'entraînement
        torch.save(model.state_dict(), "transformer_model_exemple_2.pth")
        trained = True

        # Fonction de traduction
        def translate_sentence(model, sentence, src_vocab, trg_vocab, max_len):
            model.eval()
            tokens = tokenizer(sentence.lower())
            src_indices = [src_vocab['<sos>']] + [src_vocab[token] for token in tokens] + [src_vocab['<eos>']]
            
            # Assure que le tenseur est sur le bon périphérique
            src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)

            trg_indices = [trg_vocab['<sos>']]

            for _ in range(max_len):
                trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(device)  # Tenseur cible sur le bon périphérique
                with torch.no_grad():
                    output = model(src_tensor, trg_tensor)

                next_token = output.argmax(2)[:, -1].item()
                trg_indices.append(next_token)

                if next_token == trg_vocab['<eos>']:
                    break

            translated_tokens = [trg_vocab.itos[idx] for idx in trg_indices]
            return translated_tokens

        # Traduire une nouvelle phrase
        translated_sentence = translate_sentence(model, "Transformers are powerful models", src_vocab, trg_vocab, max_len)
        print(f"Translated Sentence: {' '.join(translated_sentence)}")


        pass

    ######################"
    # Exemple 4"

    def exemple_4():
        import numpy as np
        import torch.optim as optim

        # Créez des données synthétiques pour tester le Transformer
        def create_data(num_samples, src_vocab_size, trg_vocab_size, max_len):
            data = []
            for _ in range(num_samples):
                src_seq = np.random.randint(1, src_vocab_size, size=(max_len,))
                trg_seq = np.random.randint(1, trg_vocab_size, size=(max_len,))
                data.append((src_seq, trg_seq))
            return data

        # Paramètres du modèle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src_vocab_size = 20
        trg_vocab_size = 20
        max_len = 10
        num_layers = 3
        embed_size = 256
        heads = 8
        dropout = 0.1
        forward_expansion = 4
        src_pad_idx = 0
        trg_pad_idx = 0
        learning_rate = 3e-4
        num_epochs = 2000
        batch_size = 32

        # Créez un ensemble de données de 1000 échantillons
        num_samples = 1000
        data = create_data(num_samples, src_vocab_size, trg_vocab_size, max_len)

        # Convertir la liste de données en numpy array, puis en tensor
        X_train = torch.tensor(np.array([x[0] for x in data])).to(device)
        y_train = torch.tensor(np.array([x[1] for x in data])).to(device)

        # Charger le modèle Transformer
        model = Transformer(
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            embed_size=embed_size,
            num_layers=num_layers,
            forward_expansion=forward_expansion,
            heads=heads,
            dropout=dropout,
            device=device,
            max_length=max_len
        ).to(device)

        # Définir l'optimiseur et la fonction de perte
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

        # Fonction d'entraînement
        def train_step(src, trg):
            model.train()
            optimizer.zero_grad()

            trg_input = trg[:, :-1]
            outputs = model(src, trg_input)

            # Ignorer le dernier token dans le calcul de la perte (car il n'y a pas de prédiction pour celui-ci)
            outputs = outputs.reshape(-1, outputs.shape[2])
            
            # Convertir les cibles en type long pour CrossEntropyLoss
            trg = trg[:, 1:].reshape(-1).long()

            loss = criterion(outputs, trg)
            loss.backward()
            optimizer.step()

            return loss.item()

        # Entraînement du modèle
        for epoch in range(num_epochs):
            idx = np.random.randint(0, num_samples, batch_size)
            src_batch = X_train[idx]
            trg_batch = y_train[idx]

            loss = train_step(src_batch, trg_batch)
            
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}")

        # Fonction d'évaluation (génération d'une séquence de sortie)
        def translate_sentence(model, src_sentence, max_len):
            model.eval()
            src_tensor = torch.tensor(src_sentence).unsqueeze(0).to(device)
            trg_indices = [1]  # Start with the token "<sos>"

            for i in range(max_len):
                trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(src_tensor, trg_tensor)

                next_token = output.argmax(2)[:, -1].item()
                trg_indices.append(next_token)

                if next_token == 2:  # "<eos>" token
                    break

            return trg_indices

        # Exemple de traduction avec le modèle entraîné
        src_sentence = np.random.randint(1, src_vocab_size, size=(max_len,))
        print("Source Sentence:", src_sentence)
        translated_sentence = translate_sentence(model, src_sentence, max_len)
        print("Translated Sentence:", translated_sentence)
    
    cas_applique = input("Indiquer le cas à tester (entre 1 et 4)").strip().lower()

    match cas_applique:
        case 1:
            exemple_1()
        case 2:
            exemple_2()
        case 3:
            exemple_3()
        case 4:
            exemple_4()
        
 