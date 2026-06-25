import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Any
import numpy as np



# Basic NN Module


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hiddens: List[int],
        out_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        layers = []
        for hidden_dim in hiddens:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class LSTMNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.linear(lstm_out)
        return out


# BaseNet


class BaseNet(nn.Module):
    def __init__(self, args: Any, in_dim: int, out_dim: int, module: str):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        net_type = getattr(args, f"{module}_type").lower()
        if net_type == "lstm":
            lstm_hidden = getattr(args, f"{module}_lstm_hidden")
            lstm_layers = getattr(args, f"{module}_lstm_layers")
            lstm_dropout = getattr(args, f"{module}_dropout")
            self.net = LSTMNet(
                in_dim=self.in_dim,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                out_dim=self.out_dim,
                dropout=lstm_dropout,
            )
        else:
            hiddens = getattr(args, f"{module}_hiddens")
            dropout = getattr(args, f"{module}_dropout")
            self.net = MLP(
                in_dim=self.in_dim,
                hiddens=hiddens,
                out_dim=self.out_dim,
                dropout=dropout,
                activation=nn.ReLU(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



# VAE Submodules


class S_X(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.s_clamp = args.s_clamp
        self.net = BaseNet(args, in_dim=args.feature, out_dim=args.n_cluster, module="s_x")

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x)
        logits = torch.clamp(logits, min=-self.s_clamp, max=self.s_clamp)
        probs = F.softmax(logits, dim=-1)
        return probs


class Z_S(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.n_cluster = args.n_cluster
        self.hidden_dim = args.hidden_dim
        self.mu = nn.Linear(self.n_cluster, self.hidden_dim)
        self.logsigma2 = nn.Linear(self.n_cluster, self.hidden_dim)

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        mu = self.mu(s)
        logsigma2 = self.logsigma2(s)
        return mu, logsigma2


class Z_SX(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.net = BaseNet(
            args,
            in_dim=args.feature + args.n_cluster,
            out_dim=2 * args.hidden_dim,
            module="z_sx",
        )

    def forward(self, s: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_cat = torch.cat((s, x), dim=-1)
        x_out = self.net(x_cat)
        mu = x_out[:, :, : self.hidden_dim]
        logsigma2 = x_out[:, :, self.hidden_dim :]
        return mu, logsigma2


class X_SZ(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.net = BaseNet(
            args,
            in_dim=2 * args.hidden_dim + args.n_cluster,
            out_dim=args.feature,
            module="x_sz",
        )

    def forward(self, s: Tensor, mu_z: Tensor, logsigma2_z: Tensor) -> Tensor:
        x_cat = torch.cat((s, mu_z, logsigma2_z), dim=-1)
        out = self.net(x_cat)
        return out


class X_Z(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.net = BaseNet(
            args,
            in_dim=2 * args.hidden_dim,
            out_dim=args.feature,
            module="x_z",
        )

    def forward(self, mu_z: Tensor, logsigma2_z: Tensor) -> Tensor:
        x_cat = torch.cat((mu_z, logsigma2_z), dim=-1)
        out = self.net(x_cat)
        return out






class MixtureVAE(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.eps = getattr(args, "eps", 1e-8)
        self.logvar_min = getattr(args, "logvar_min", -10.0)
        self.logvar_max = getattr(args, "logvar_max", 10.0)

        self.s_x = S_X(args)
        self.z_s = Z_S(args)
        self.z_sx = Z_SX(args)

        if self.args.reconstruction_on_s:
            self.x_sz = X_SZ(args)
        else:
            self.x_z = X_Z(args)

        self.loss = torch.tensor(0.0)
        self.loss_components = {}

    def forward(self, x: Tensor) -> Tensor:
        #cluster probabilities
        s_prob = self.s_x(x)  # [B, T, K]
        s_prob_safe = torch.clamp(s_prob, min=self.eps, max=1.0)

        #gumbel-softmax sampling
        s = F.gumbel_softmax(
            torch.log(s_prob_safe),
            tau=self.args.tau,
            hard=self.args.hard,
            dim=-1,
        )

        #latent parameters
        mu_z_q, logsigma2_z_q = self.z_sx(s, x)
        mu_z_p, logsigma2_z_p = self.z_s(s)

        logsigma2_z_q = torch.clamp(logsigma2_z_q, min=self.logvar_min, max=self.logvar_max)
        logsigma2_z_p = torch.clamp(logsigma2_z_p, min=self.logvar_min, max=self.logvar_max)

        #reconstruction
        if self.args.reconstruction_on_s:
            if self.args.reconstruction_on_z == "q":
                out = self.x_sz(s, mu_z_q, logsigma2_z_q)
            elif self.args.reconstruction_on_z == "p":
                out = self.x_sz(s, mu_z_p, logsigma2_z_p)
            else:
                raise ValueError("Invalid value for reconstruction_on_z. Expected 'q' or 'p'.")
        else:
            if self.args.reconstruction_on_z == "q":
                out = self.x_z(mu_z_q, logsigma2_z_q)
            elif self.args.reconstruction_on_z == "p":
                out = self.x_z(mu_z_p, logsigma2_z_p)
            else:
                raise ValueError("Invalid value for reconstruction_on_z. Expected 'q' or 'p'.")

        #losses
        loss_i = information_loss(s_prob_safe, self.args, eps=self.eps)
        loss_t = transition_loss(s_prob_safe, self.args)
        loss_m = mixture_loss(
            mu_z_q,
            logsigma2_z_q,
            mu_z_p,
            logsigma2_z_p,
            self.args,
            eps=self.eps,
        )
        loss_r = reconstruction_loss(out, x, self.args)
        loss_b = regime_balance_loss(s_prob_safe, self.args, eps=self.eps)

        if self.args.loss_clamp is not None:
            loss_i = torch.clamp(loss_i, min=-self.args.loss_clamp, max=self.args.loss_clamp)
            loss_t = torch.clamp(loss_t, min=-self.args.loss_clamp, max=self.args.loss_clamp)
            loss_m = torch.clamp(loss_m, min=-self.args.loss_clamp, max=self.args.loss_clamp)
            # Do not clamp loss_b: it is non-negative and interpretable.

        lamda_b = getattr(self.args, "lamda_b", 0.0)

        total_loss = (
            loss_r
            + self.args.lamda_m * loss_m
            + self.args.lamda_i * loss_i
            + self.args.lamda_t * loss_t
            + lamda_b * loss_b
        )

        # Store unweighted loss components for validation diagnostics.
        self.loss_components = {
            "loss_r": loss_r.detach(),
            "loss_m": loss_m.detach(),
            "loss_i": loss_i.detach(),
            "loss_t": loss_t.detach(),
            "loss_b": loss_b.detach(),
            "total_loss": total_loss.detach(),
        }

        if not torch.isfinite(total_loss):
            raise ValueError(
                f"Non-finite loss detected. "
                f"loss_r={loss_r.item()}, loss_m={loss_m.item()}, "
                f"loss_i={loss_i.item()}, loss_t={loss_t.item()}, "
                f"loss_b={loss_b.item()}"
            )

        self.loss = total_loss
        return out

    def get_s_prob(self, x: Tensor) -> Tensor:
        s_prob = self.s_x(x)
        s_prob = torch.clamp(s_prob, min=self.eps, max=1.0)
        return s_prob

    def get_z(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        s_prob = self.get_s_prob(x)
        mu_z_q, logsigma2_z_q = self.z_sx(s_prob, x)
        logsigma2_z_q = torch.clamp(logsigma2_z_q, min=self.logvar_min, max=self.logvar_max)
        return mu_z_q, logsigma2_z_q



# Loss Functions


def information_loss(s_prob: Tensor, args, eps: float = 1e-8) -> Tensor:
    s_prob_safe = torch.clamp(s_prob, min=eps, max=1.0)
    loss = s_prob_safe * torch.log(s_prob_safe)

    if args.loss_mode == "sum":
        return loss.sum()
    elif args.loss_mode == "mean":
        return loss.mean()
    elif args.loss_mode == "norm":
        return loss.sum()
    else:
        raise ValueError("Invalid loss_mode. Must be 'sum', 'mean', or 'norm'.")


def mixture_loss(
    mu_z_q: Tensor,
    logsigma2_z_q: Tensor,
    mu_z_p: Tensor,
    logsigma2_z_p: Tensor,
    args,
    eps: float = 1e-8,
) -> Tensor:
    logsigma2_z_q = torch.clamp(logsigma2_z_q, min=-10.0, max=10.0)
    logsigma2_z_p = torch.clamp(logsigma2_z_p, min=-10.0, max=10.0)

    ln2 = torch.log(torch.tensor(2.0, device=mu_z_q.device))
    term0 = 0.5 * (logsigma2_z_p - logsigma2_z_q)
    term1 = torch.exp(logsigma2_z_q - logsigma2_z_p - 2 * ln2)
    term2 = (mu_z_q - mu_z_p) ** 2 / (2 * torch.exp(logsigma2_z_p) + eps)
    loss = term0 + term1 + term2

    if args.loss_mode == "sum":
        return loss.sum()
    elif args.loss_mode == "mean":
        return loss.mean()
    elif args.loss_mode == "norm":
        return loss.sum() / np.sqrt(term0.size(-1))
    else:
        raise ValueError("Invalid loss_mode. Must be 'sum', 'mean', or 'norm'.")


def reconstruction_loss(out: Tensor, x: Tensor, args) -> Tensor:
    loss = (out - x) ** 2

    if args.loss_mode == "sum":
        return loss.sum()
    elif args.loss_mode == "mean":
        return loss.mean()
    elif args.loss_mode == "norm":
        return loss.sum() / np.sqrt(x.size(-1))
    else:
        raise ValueError("Invalid loss_mode. Must be 'sum', 'mean', or 'norm'.")


def regime_balance_loss(s_prob: Tensor, args, eps: float = 1e-8) -> Tensor:
    """
    Discourages latent-state collapse by penalizing degenerate average
    posterior regime usage.

    Shape expected: s_prob = [batch, time, n_cluster].

    The KL term is scaled under loss_mode='sum' so lamda_b is not negligible
    compared with the summed reconstruction/regularization losses.
    """
    s_prob_safe = torch.clamp(s_prob, min=eps, max=1.0)

    avg_usage = s_prob_safe.mean(dim=(0, 1))  # [K]
    avg_usage = avg_usage / avg_usage.sum().clamp_min(eps)

    n_cluster = avg_usage.size(0)
    uniform = torch.full_like(avg_usage, 1.0 / n_cluster)

    kl_uniform_to_usage = torch.sum(
        uniform * (torch.log(uniform + eps) - torch.log(avg_usage + eps))
    )

    if args.loss_mode == "sum":
        return kl_uniform_to_usage * s_prob.size(0) * s_prob.size(1)
    elif args.loss_mode == "mean":
        return kl_uniform_to_usage
    elif args.loss_mode == "norm":
        return kl_uniform_to_usage * s_prob.size(0)
    else:
        raise ValueError("Invalid loss_mode. Must be 'sum', 'mean', or 'norm'.")


def transition_loss(s_prob: Tensor, args) -> Tensor:
    if args.transition != "jump":
        raise ValueError("Only 'jump' transition is currently supported.")

    if hasattr(args, "jump_mx") and args.jump_mx is not None:
        jump_mx = torch.tensor(args.jump_mx, dtype=torch.float32, device=s_prob.device)
    else:
        n = s_prob.size(-1)
        jump_mx = torch.ones(n, n, device=s_prob.device) - torch.eye(n, device=s_prob.device)
        jump_mx = jump_mx.float()

    s0 = s_prob[:, :-1, :]
    s1 = s_prob[:, 1:, :]
    p = torch.matmul(s0.unsqueeze(-1), s1.unsqueeze(-2))
    p = (p * jump_mx).sum(dim=(-1, -2))

    if args.loss_mode == "sum":
        return p.sum()
    elif args.loss_mode == "mean":
        return p.mean()
    elif args.loss_mode == "norm":
        denom = max(s_prob.size(0) * max(s_prob.size(1) - 1, 1), 1)
        return p.sum() / denom
    else:
        raise ValueError("Invalid loss_mode. Must be 'sum', 'mean', or 'norm'.")
