import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random

#### DATA GENERATION ####
def generate_time_series_dataset(O, F, P, T):
    t = np.arange(T)
    sig = np.zeros((O, T), dtype=np.float32)
    F = np.array(F)
    P = np.array(P)
    phases = 2 * np.pi * np.random.rand(O, len(F))
    for o in range(O):
        for k, freq in enumerate(F):
            sig[o] += np.sqrt(P[o, k]) * np.sin(2 * np.pi * freq * t / T + phases[o, k])
    return sig.T

def build_dataset(O, F_list, P_list, T, S, sigma, split=(0.8, 0.2), seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    num_classes = 5
    all_X, all_Y = [], []
    for c in range(num_classes):
        F, P = F_list[c], P_list[c]
        for s in range(S):
            sig = generate_time_series_dataset(O, F, P, T)
            sig += np.random.normal(0, sigma, sig.shape)
            all_X.append(sig)
            all_Y.append(c)
    all_X, all_Y = np.stack(all_X), np.array(all_Y)
    idx = np.arange(len(all_Y))
    np.random.shuffle(idx)
    all_X, all_Y = all_X[idx], all_Y[idx]
    n_train = int(len(all_Y)*split[0])
    X_train, Y_train = all_X[:n_train], all_Y[:n_train]
    X_test, Y_test = all_X[n_train:], all_Y[n_train:]
    X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)
    X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)
    train_ds = TensorDataset(X_train, Y_train)
    test_ds = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    return train_loader, test_loader

#### MODEL ####
class RNNClassifierReconstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, nonlinearity='tanh'):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1,
                          nonlinearity=nonlinearity, batch_first=True)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.recon_head = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        out, h_n = self.rnn(x)
        last_hidden = out[:, -1, :]
        logits = self.class_head(last_hidden)
        recon_out = self.recon_head(out)
        return logits, recon_out, out

#### TRAINING ####
def train_rnn_reconstruction_time(model, train_loader, test_loader, 
                                  epochs=10, alpha=1.0, reconstruction_time=0, device='cpu'):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    for ep in range(epochs):
        model.train()
        total, correct, total_loss = 0, 0, 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            logits, recon, _ = model(X)
            loss = ce_loss(logits, Y)
            # Reconstruction loss with time shift
            shift = int(reconstruction_time)
            if alpha > 0:
                if shift == 0:
                    target = X
                elif shift < 0: # reconstruct the past from present
                    pad = torch.zeros_like(X[:, :abs(shift), :])
                    target = torch.cat([pad, X[:, :X.shape[1]+shift, :]], dim=1)
                else: # shift > 0, i.e. reconstruct the future from present
                    pad = torch.zeros_like(X[:, -shift:, :])
                    target = torch.cat([X[:, shift:, :], pad], dim=1)
                loss = loss + alpha * mse_loss(recon, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = logits.argmax(dim=1)
            total += X.size(0)
            correct += (pred == Y).sum().item()
            total_loss += loss.item() * X.size(0)
        train_accs.append(correct / total)
        train_losses.append(total_loss / total)
        # Evaluate test
        model.eval()
        total, correct, total_loss = 0, 0, 0
        with torch.no_grad():
            for X, Y in test_loader:
                X, Y = X.to(device), Y.to(device)
                logits, recon, _ = model(X)
                loss = ce_loss(logits, Y)
                shift = int(reconstruction_time)
                if alpha > 0:
                    if shift == 0:
                        target = X
                    elif shift < 0:
                        pad = torch.zeros_like(X[:, :abs(shift), :])
                        target = torch.cat([pad, X[:, :X.shape[1]+shift, :]], dim=1)
                    else:
                        pad = torch.zeros_like(X[:, -shift:, :])
                        target = torch.cat([X[:, shift:, :], pad], dim=1)
                    loss = loss + alpha * mse_loss(recon, target)
                pred = logits.argmax(dim=1)
                total += X.size(0)
                correct += (pred == Y).sum().item()
                total_loss += loss.item() * X.size(0)
            test_accs.append(correct / total)
            test_losses.append(total_loss / total)
    # Return the final epoch's results
    return train_accs[-1], test_accs[-1], train_losses[-1], test_losses[-1]

#### EXPERIMENT LOOP ####
if __name__ == "__main__":
    # Settings
    O, T, S = 2, 500, 20
    sigma = 0.3
    F_list = [ [2,8,16],[2,6,12],[3,9,15],[1,5,14],[4,10,18] ]
    P_list = [
        np.array([[5,1,1],[2,2,6]]), np.array([[4,3,1],[1,5,2]]),
        np.array([[7,2,2],[2,1,7]]), np.array([[3,4,2],[4,2,3]]),
        np.array([[2,2,6],[6,1,2]])
    ]
    train_loader, test_loader = build_dataset(O, F_list, P_list, T, S, sigma)
    input_dim, hidden_dim, num_classes = O, 128, 5
    # Experiment parameters
    reconstruction_times = np.linspace(-T//2, T//2, 21, dtype=int)  # 21 values from -250 to +250
    alpha = 1.0   # weight for reconstruction loss; set to 0.0 for classification-only

    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    for reconstruction_time in reconstruction_times:
        print(f"Running for reconstruction_time = {reconstruction_time}")
        # Fresh model each time
        model = RNNClassifierReconstructor(input_dim, hidden_dim, num_classes).to('cpu')
        acc_train, acc_test, loss_train, loss_test = train_rnn_reconstruction_time(
            model, train_loader, test_loader, epochs=10, 
            alpha=alpha, reconstruction_time=reconstruction_time, device='cpu')
        train_accs.append(acc_train)
        test_accs.append(acc_test)
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        print(f"...train_acc={acc_train:.3f} test_acc={acc_test:.3f} train_loss={loss_train:.3f} test_loss={loss_test:.3f}")

    # Plot results
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(reconstruction_times, train_accs, 'o-', label='Train Acc')
    plt.plot(reconstruction_times, test_accs, 'o-', label='Test Acc')
    plt.xlabel('Reconstruction Time')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Reconstruction Time')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(reconstruction_times, train_losses, 'o-', label='Train Loss')
    plt.plot(reconstruction_times, test_losses, 'o-', label='Test Loss')
    plt.xlabel('Reconstruction Time')
    plt.ylabel('Loss')
    plt.title('Loss vs Reconstruction Time')
    plt.legend()
    plt.tight_layout()
    plt.show()