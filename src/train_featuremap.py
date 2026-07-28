import torch
import torch.nn as nn

def train_featuremap(f, x, y, proj, max_epoch=3000, print_freq=500):
    optimizer = torch.optim.Adam(f.parameters())
    optimizer.zero_grad()
    with torch.no_grad():
        x_red = x@proj
    for epoch in range(1,max_epoch+1):
        f.zero_grad()

        y_pred = f(x_red).squeeze()
        loss = nn.MSELoss()(y, y_pred)
        if (epoch%print_freq)==0 :
            cur_loss = loss.item()
            print(f"loss: {loss:>6f}  [{epoch:>5d}/{max_epoch:>5d}]")

        loss.backward()
        optimizer.step()
        epoch +=1
    return loss.item(), f
