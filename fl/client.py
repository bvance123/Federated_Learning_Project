"""Client side: local training and personal models with mu.

Owner: Nikolay
"""

import torch
from torch import nn

from fl.model import Model,get_weights, set_weights



def predict(model_weights, X):
    """Run patients through a model, return P(disease)."""

    #Define the model

    #Define the global model from Bennett's class
    model=Model()

    #Load the model weights into Bennett's global model
    model.load_state_dict(model_weights)

    #Set to prediction mode. No training is performed
    model.eval()

    #Stop tracking gradients since we are not training
    with torch.no_grad():
        #This runs patients through model to actually make predictions
        return model(X)



def local_update(model, global_weights, train_data, mu, epochs, lr=0.01, batch_size=32, init_weights=None):
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

    # init_weights lets the model start from custom weights instead of the
    # global weights. This is for later in the simulation, when this method
    # is called after the first run and we want to continue from a previously
    # established personal model. The leash below still pulls toward the
    # GLOBAL weights, whichever starting point is used.
    if init_weights:
        start=init_weights
    else:
        start=global_weights

    for name in start:
        cloned[name]=start[name].clone()
    model.load_state_dict(cloned)
    model.train()


    # 2. Frozen reference copy for the leash term
    global_params=[]
    for name,p in model.named_parameters():
        global_params.append(global_weights[name].detach().clone())

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





