import pickle
import torch
import gpt_v2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Character mapping setup
chars = ""

with open("vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s if c in string_to_int]
decode = lambda l: ''.join([int_to_string[i] for i in l])

model = gpt_v2.GPTLanguageModel(vocab_size)

try:
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
        print('loaded successfully!')
except:
    print('Error: Unable to load previous model parameters')

model = model.to(device)


prompt = ""
while prompt != "exit":
    prompt = input("Prompt (type 'exit' to terminate program): ")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=250)[0].tolist())
    print(generated_chars)