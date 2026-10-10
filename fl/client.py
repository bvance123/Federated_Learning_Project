"""Client side: local training and personal models with mu.

Owner: Nikolay
"""
import torch
from torch import nn
from fl.model import get_weights


def local_update(model, global_weights, train_data, mu, epochs, lr=0.01, batch_size=32):
    """Train one hospital's model starting from the global weights.

    model:          fl.model.Model (a fresh copy; safe to modify)
    global_weights: state_dict of the current global model
    train_data:     (X, y) tensors from load_center: X float [n, 13], y float [n] in {0, 1}
    mu:             drift-penalty strength; loss = BCE + (mu / 2) * ||w - w_global||^2
    epochs:         number of local epochs
    returns:        updated state_dict (use fl.model.get_weights(model))

    Sanity checks:
      mu = 0       -> same as plain local training
      mu very big  -> stays almost identical to global_weights
    """
    # TODO (Nikolay)
   
    # 1. Copy the global weights into the model
    cloned={}
    for name in global_weights:
        cloned[name]=global_weights[name].clone()
    model.load_state_dict(cloned)
    model.train()


    # 2. Frozen reference copy for the leash term
    global_params=[]
    for p in model.parameters():
        global_params.append(p.detach().clone())

    # 3. Training loop
    X, y = train_data
    n = X.shape[0]
    loss_fn = nn.BCELoss()

    #object which actually changes the weights
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):

        #Randomly shuffle patient indeces after each epoch
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()

            #Make predictions
            probs = model(X[idx])

            #Define loss function
            loss = loss_fn(probs, y[idx])

            if mu > 0:
                params = list(model.parameters())
                prox = 0
                for j in range(len(params)):

                    #Find difference between local and global model
                    diff = params[j] - global_params[j]

                    # diff: adds up the numbers inside ONE tensor (the 13 weights, or the 1 bias)
                     #prox:adds those per-tensor totals together
                    prox = prox + (diff ** 2).sum()

                #Add mu to loss function
                loss = loss + (mu / 2) * prox

            loss.backward()
            opt.step()

        
    return get_weights(model)





