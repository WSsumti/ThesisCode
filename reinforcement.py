import torch as torch
from utilss import Utilss

class Env:
    def __init__(self):
        self.done = False
        self.current_step = 0

    def reset(self, fold, train_df, tokenizer):
        self.current_step = 1
        self.done = False
        #print(f"## Fold #{fold}")
        train_loader, valid_loader = Utilss.prepare_loaders(train_df, fold=fold, tokenizer=tokenizer)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.current_item = next(iter(train_loader))
        return self.current_item

    def step(self, action, device):
        self.current_step += 1
        rewards = []
        correct = 0
        targets = self.current_item['targets'].to(device)
        #         correct += torch.sum(action == targets)
        #         reward = correct / len(action)
        for eA, eC in zip(action, targets):
            if eA.item() == eC.item():
                rewards.append(1)
                correct += 1
            else:
                rewards.append(-1)
        self.current_item = next(iter(self.train_loader))
        if self.current_step > len(self.train_loader):
            self.done = True
        return self.current_item, rewards, torch.tensor(correct)
    

def custom_policy_loss(log_probs, returns, gamma, ascent=True):
    # Compute discounted rewards
    discounted_returns = []
    R = 0
    if (ascent == True):
        for r in returns[::-1]:
            R = r + gamma * R
            discounted_returns.insert(0, R)
    else:
        for r in returns[::-1]:
            R = r - gamma * R
            discounted_returns.insert(0, R)
    discounted_returns = torch.tensor(discounted_returns)

    # Normalize rewards
    discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

    # Compute the loss
    policy_loss = []
    for log_prob, R in zip(log_probs, discounted_returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.stack(policy_loss).sum()

    return policy_loss
