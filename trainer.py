#Pete Zimmerman
#written with assistance from claude
#Jun-2026
#Contains: training loop for sin approximation
#purpose: exploratory study on MLP hyperparameter optimization

import torch
import torch.nn.functional as F
import numpy as np
import net

LR = 1e-3
EPOCHS = 5000
BATCH_SIZE = 64
def train(params: list[int]) -> list[float]:
    model = net.build(params)()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []

    for _ in range(EPOCHS):
        x = torch.FloatTensor(BATCH_SIZE, 1).uniform_(-2 * np.pi, 2 * np.pi)
        y = torch.FloatTensor(np.sin(x.numpy()))

        pred = model(x)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses
