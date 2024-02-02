import numpy as np
import torch
import torch.nn as nn

from model.logreg import LogReg
from utils import dataloader
from model import DGI
import wandb

lr = 0.001
hid_units = 512
epochs = 100
batch_size = 1
patience = 20

nonlinearity = 'prelu'
dataset_name = "cora"
# dataset_name = "citeseer"
# dataset_name = "PubMed"

# Load dataset
A, features, labels, idx_train, idx_val, idx_test = dataloader.load_data_cite(dataset_name)
func = "average"
# func = "diffpool"
# func = "sum_norm"

pname = "ML4Grpahs_DGI_"
project_name = pname + dataset_name +"_"+ func +"_"+ str(epochs)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

# normalize the features and adj
adj = dataloader.get_A_hat_torch(A)
features = dataloader.normalize_features(features)

labels = torch.FloatTensor(labels[np.newaxis])

model = DGI(nb_nodes,ft_size, hid_units)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
best = 1e9
best_t = 0

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="ML4Grpahs_DGI",
    name=project_name,
    # track hyperparameters and run metadata
    config={
        "model_learning_rate": lr,
        "architecture": "GCN",
        "dataset": dataset_name,
        "epochs": epochs
    }
)
for epoch in range(epochs):
    model.train()
    optimiser.zero_grad()

    # generate negative samples
    # corruption function
    idx = np.random.permutation(nb_nodes)
    negative_features = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    logits = model(features, negative_features, adj, func)

    loss = b_xent(logits, lbl)
    print('Epoch:', epoch, 'Loss:', loss.item())
    wandb.log({"loss_training": loss.item()})
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()


print('Loading {}th epoch'.format(best_t))



# test accuracy of the best performance model weight
model.load_state_dict(torch.load('best_dgi.pkl'))
embeds, _ = model.embed(features, adj, func)
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

tot = torch.zeros(1)
# tot = tot.cuda()

accs = []
for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    # log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    # best_acc = best_acc.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    wandb.log({"acc_test": acc})
    print(acc)
    tot += acc

print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print("accs mean:", accs.mean())
print("accs std:", accs.std())

wandb.finish()


