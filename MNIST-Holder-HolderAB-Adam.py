####################################################################
# Holder and acceleration bound, Holder, Adam
# MNIST benchmark
# by Marco Gori
####################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Hyperparameters
batch_size = 128
epochs = 15
beta = 0.75
eta_beta = 1
eta_adam = 0.001
acc_threshold = 0.5
acc_threshold = 0.1
eps = 1e-12

# Data
transform = transforms.ToTensor()

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Evaluation
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred==target).sum().item()
            total += target.size(0)

    return correct/total


# Hölder method
def train_holder(model):

    prev_velocity = [torch.zeros_like(p) for p in model.parameters()]
    test_acc_hist = []

    for epoch in range(epochs):

        for data,target in train_loader:

            data,target = data.to(device),target.to(device)

            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out,target)
            loss.backward()

            grad_norm_sq = sum(torch.sum(p.grad**2) for p in model.parameters() if p.grad is not None)

            scale = (loss.item()**beta)/(grad_norm_sq.item()+eps)

            with torch.no_grad():

                for i,p in enumerate(model.parameters()):

                    if p.grad is None:
                        continue

                    velocity = -eta_beta*scale*p.grad
                    p += velocity
                    prev_velocity[i] = velocity.clone()

        test_acc = evaluate(model)
        test_acc_hist.append(test_acc)

        print("Holder Epoch",epoch+1,"Test acc",test_acc)

    return test_acc_hist


# Acceleration limited Holder
def train_holder_acc(model):

    prev_velocity = [torch.zeros_like(p) for p in model.parameters()]
    test_acc_hist = []

    for epoch in range(epochs):

        for data,target in train_loader:

            data,target = data.to(device),target.to(device)

            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out,target)
            loss.backward()

            grad_norm_sq = sum(torch.sum(p.grad**2) for p in model.parameters() if p.grad is not None)

            scale = (loss.item()**beta)/(grad_norm_sq.item()+eps)

            with torch.no_grad():

                for i,p in enumerate(model.parameters()):

                    if p.grad is None:
                        continue

                    velocity = -eta_beta*scale*p.grad

                    acc = torch.norm(velocity-prev_velocity[i])

                    if acc > acc_threshold: # switch to limit stiffness

                        velocity = prev_velocity[i] + (velocity-prev_velocity[i])*(acc_threshold/(acc+eps))

                    p += velocity
                    prev_velocity[i] = velocity.clone()

        test_acc = evaluate(model)
        test_acc_hist.append(test_acc)

        print("Holder-Acc Epoch",epoch+1,"Test acc",test_acc)

    return test_acc_hist


# Adam training
def train_adam(model):

    optimizer = optim.Adam(model.parameters(),lr=eta_adam)
    test_acc_hist=[]

    for epoch in range(epochs):

        for data,target in train_loader:

            data,target = data.to(device),target.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out,target)
            loss.backward()
            optimizer.step()

        test_acc=evaluate(model)
        test_acc_hist.append(test_acc)

        print("Adam Epoch",epoch+1,"Test acc",test_acc)

    return test_acc_hist


# Shared initialization
base_model = Net().to(device)

model_holder = Net().to(device)
model_adam   = Net().to(device)
model_acc    = Net().to(device)

model_holder.load_state_dict(base_model.state_dict())
model_adam.load_state_dict(base_model.state_dict())
model_acc.load_state_dict(base_model.state_dict())

# Training
acc_holder = train_holder(model_holder)
acc_adam   = train_adam(model_adam)
acc_acc    = train_holder_acc(model_acc)

# Plot
plt.figure()

plt.plot(acc_holder,label="Holder")
plt.plot(acc_adam,label="Adam")
plt.plot(acc_acc,label="Holder + acceleration bound")

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.show()
