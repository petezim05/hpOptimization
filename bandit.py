import argparse
import csv
import os
import net
import trainer
import torch
import torch.nn.functional as F
import numpy
import math
from tokenizer import ArchTokenizer

LOSSES_CSV = "losses.csv"

def append_run_to_csv(losses: list[float], params: list[int]) -> None:
    layer_str = " -> ".join(str(w) for w in params)
    arch = f"1 -> {layer_str} -> 1"
    needs_header = not os.path.exists(LOSSES_CSV)
    with open(LOSSES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            header = ["architecture", "lr", "epochs", "batch_size"] + [f"epoch_{i+1}" for i in range(len(losses))]
            writer.writerow(header)
        writer.writerow([arch, trainer.LR, trainer.EPOCHS, trainer.BATCH_SIZE] + losses)

seq_len = 6
VOCAB_SIZE = 513
EOS = 513
tokenizer = ArchTokenizer()

#utilities
def temp(episode):
    return 2.0 * math.exp(-1 * (episode / 10000))

#gain functions
def logGains(state, prevState):
    return -1 * math.log10(state / prevState)

def normGains(state, prevState):
    return ((prevState - state) / prevState)

#assumes autoregressive model, must be only ints in list
def valid(arr:list):
    if(EOS not in arr): return False
    if(len(arr) < 2): return False
    for i in range(1,len(arr)):
        if(arr[i-1] == 0 or arr[i-1] == EOS): return False
        if(arr[i] == EOS): break
    return True

#G(S_t, S_t-1)
invalidModel = -1
def gain(state:float, prevState:float, mode:str, model:list):
    if(mode == 'log' or not mode):
        g = logGains
    elif(mode == 'norm'):
        g = normGains
    if(not valid(model)): return invalidModel
    return g(state, prevState)

#state functions
#AUC of loss curve as state signal
def AUC(points: list):
    return numpy.trapz(points)

#MLP controller: takes partial sequence, outputs logits over next token
class AutoregressiveModelBuilder(torch.nn.Module):
    def __init__(self, seq_len, vocab_size=VOCAB_SIZE):
        super().__init__()

        params = [128, 128]
        layers = []
        depth = len(params)

        for i in range(0, depth):
            if(i == 0):
                layers.append(torch.nn.Linear(seq_len, params[i]))
            else:
                layers.append(torch.nn.Linear(params[i-1], params[i]))
        layers.append(torch.nn.Linear(params[-1], vocab_size))

        self.layers = torch.nn.ModuleList(layers)
        self.temperature = 1.0

    def forward(self, x):
        for i in range(0, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)

    def generate(self):
        sequence = [1] * seq_len
        log_probs = []

        for pos in range(seq_len):
            x = torch.FloatTensor(sequence)
            logits = self.forward(x)
            if self.temperature == 0:
                idx = torch.argmax(logits)
            else:
                logits = logits / self.temperature
                idx = torch.distributions.Categorical(logits=logits).sample()
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(idx))
            token = idx.item() + 1  # logit index 0-1024 -> token value 1-1025
            sequence[pos] = token
            if token == EOS:
                break

        return tokenizer.decode(torch.tensor(sequence)), log_probs

#the bandit finally arrives
#bias eos initially to ensure valid models are produced
hermes = AutoregressiveModelBuilder(seq_len)
hermes.layers[-1].bias.data[EOS - 1] = 3.0 

LR = 1e-3
EPISODES = 5000
BATCH_SIZE = 8
EVALS_PER_MODEL = 3

def evaluate(params: list) -> float:
    aucs = []
    for _ in range(EVALS_PER_MODEL):
        losses = trainer.train(params)
        append_run_to_csv(losses, params)
        aucs.append(float(AUC(losses)))
    return float(numpy.mean(aucs))

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 100

def trainAgent(verbose=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    optimizer = torch.optim.Adam(hermes.parameters(), lr=LR)
    prevState = None
    best = (float('inf'), [])

    for episode in range(EPISODES):
        hermes.temperature = temp(episode)

        log_prob_sums = []
        gains = []
        states = []

        for _ in range(BATCH_SIZE):
            params, log_probs = hermes.generate()
            encoded = tokenizer.encode(params).tolist()

            if not valid(encoded):
                continue

            state = evaluate(params)
            states.append(state)

            if state < best[0]:
                best = (state, params)

            if prevState is None:
                continue

            g = gain(state, prevState, 'log', encoded)
            if g == invalidModel:
                continue

            gains.append(g)
            log_prob_sums.append(torch.stack(log_probs).sum())

        if states:
            prevState = float(numpy.mean(states))

        if not gains:
            continue

        gains_t = torch.tensor(gains, dtype=torch.float32)
        if len(gains_t) > 1 and gains_t.std() > 0:
            gains_t = (gains_t - gains_t.mean()) / (gains_t.std() + 1e-8)

        optimizer.zero_grad()
        torch.stack([-lp * g for lp, g in zip(log_prob_sums, gains_t)]).mean().backward()
        optimizer.step()

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(hermes.state_dict(), os.path.join(CHECKPOINT_DIR, f"hermes_ep{episode + 1}.pt"))

        if verbose:
            arch = "1 -> " + " -> ".join(str(w) for w in best[1]) + " -> 1"
            print(f"episode {episode:4d} | valid={len(gains)} | prevState={prevState:.4f} | temp={hermes.temperature:.4f} | best={best[0]:.4f} {arch}")

    if verbose:
        arch = "1 -> " + " -> ".join(str(w) for w in best[1]) + " -> 1"
        summary = (
            f"=== Bandit Run Summary ===\n"
            f"Episodes:     {EPISODES}\n"
            f"Batch size:   {BATCH_SIZE}\n"
            f"Evals/model:  {EVALS_PER_MODEL}\n"
            f"\n"
            f"Best architecture: {arch}\n"
            f"Best AUC:          {best[0]:.6f}\n"
        )
        print(summary)
        with open("results.txt", "w") as f:
            f.write(summary)

    return best

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='print episode summaries and write results.txt')
    args = parser.parse_args()
    trainAgent(verbose=args.verbose)
