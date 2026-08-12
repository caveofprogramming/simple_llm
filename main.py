import os
import json
import model as m

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


if __name__ == '__main__':
    main()