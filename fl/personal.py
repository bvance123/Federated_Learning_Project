from fl.client import local_update, predict
from fl.model import Model, get_weights
from fl.server import aggregate, drift
from eval.metrics import auroc
import math

def run_personal(clients, local_val_auc, n_rounds, epochs,mu_start=0.1, warmup=5):

    # clients: data from different hospitals: list of {"name", "train": (X, y), "val": (X, y)}
    # local_val_auc:local hospitals are one without leash and consist of 
    #  {name: validation AUROC of that hospital's local-only model}


    #Get weights and return as a dictionary of tensors
     global_w = get_weights(Model())

     v = {}
     mu = {}
     history = []

     #Iterate over all hospitals
     for client in clients:
        name=client["name"]
        copy_of_global = {}
        mu[name] = mu_start

        #Copy global initialization into all clients
        for k,t in global_w.items():
          copy_of_global[k] = t.clone()
        v[name] = copy_of_global

    
     for r in range(n_rounds):

        

        #Contains the actual weights of each hospital model
        updates=[]

        #Impacts what weight a hospital will have on global model
        counts=[]

       
        global_auc = {}

        #This is FedAvg without a leash
        for client in clients:
            #Create local weights
            w = local_update(Model(), global_w, client["train"], mu=0, epochs=epochs)
            updates.append(w)
            counts.append(len(client["train"][1]))

        #Creates global model based on weighted avg from patients
        global_w = aggregate(updates, counts)

        
        #Score the new global model on each hospital's VALIDATION data
        for client in clients:
            name = client["name"]
            X_val, y_val = client["val"]
            probs = predict(global_w, X_val)
            global_auc[name] = auroc(y_val, probs)


        #Adjust mu
        # Global at least as good as local-only: double mu (shorter leash).
        # Otherwise halve mu (longer leash).
        for client in clients:
            name = client["name"]
            if r>=warmup and not math.isnan(global_auc[name]) and not math.isnan(local_val_auc[name]):
                if global_auc[name]>=local_val_auc[name]:
                 mu[name]=2*mu[name]
                else:
                 mu[name]=mu[name]/2
                mu[name] = min(max(mu[name], 0.01), 1.0)

        #Personal models. These are NEVER averaged.
        #Notice that the local models do not change mu, they
        # just set it for the persona models 
        for client in clients:
             name = client["name"]
             v[name] = local_update(Model(), global_w, client["train"],
                                mu=mu[name], epochs=epochs, init_weights=v[name])
           
        #Record everything important for that round
        for client in clients:
            name = client["name"]
            row = {
             "round": r,
             "hospital": name,
              "mu": mu[name],
             "global_auc": global_auc[name],
             "local_val_auc": local_val_auc[name],
              "drift": drift(v[name], global_w),
              }
            history.append(row)

     return global_w, v, mu, history