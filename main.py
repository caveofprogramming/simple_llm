import os
import json
import model as m
import torch
import torch.nn.functional as F

def load_config(file):
    with open(file, 'r') as f:
        return json.load(f)

def load_data(dir):
    files = os.listdir(dir)

    text = []

    for file in files:
        with open(os.path.join(dir, file), 'r') as f:
            text.append(f.read())

    return "".join(text)

def train(model, loader, context_len, batch_size, steps, lr=3e-4):
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(steps):
        x, y = loader.get_batch(context_len, batch_size)
        logits = model(x)
        vocab_size = logits.shape[-1]
        logits = logits.view(-1, vocab_size)
        y = y.view(-1)

        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} loss {loss.item():.4f}")

def generate(model, vocab, context, max_new_tokens=200):
    ids = vocab.encode(context).tolist()

    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.pos_embed.weight.shape[0]:]])

        logits = model(x)
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)

    return vocab.decode(torch.tensor(ids))


def main():
    config = load_config('config.json')
    text = load_data(config['data_dir'])

    vocab = m.Vocab(text)
    dataset = m.CharDataset(text, vocab)
    loader = m.BatchLoader(dataset)

    context_len = config['context_len']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']

    model = m.TransformerModel(
        vocab_size=vocab.vocab_size,
        context_len=context_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )

    steps = config['steps']
    batch_size = config['batch_size']
    lr = config["learning_rate"]

    train(model, loader, context_len, batch_size, steps, lr)

    output = generate(model, vocab, "Once upon a time", max_new_tokens=config['max_new_tokens'])
    print(output)


if __name__ == '__main__':
    main()