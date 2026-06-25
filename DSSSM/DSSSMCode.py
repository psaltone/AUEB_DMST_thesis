import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class DSSSM(nn.Module):

    def __init__(
        self,
        x_dim,
        y_dim,
        h_dim,
        z_dim,
        d_dim,
        n_layers,
        device,
        bidirection=False,
        bias=False,
        dataname=None,
        lamda_b=0.0,
        lamda_entropy=0.0
    ):
        super(DSSSM, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.d_dim = d_dim
        self.n_layers = n_layers
        self.device = device
        self.temperature = 0.5
        self.bidirection = bidirection
        self.dataname = dataname
        self.lamda_b = float(lamda_b)
        self.lamda_entropy = float(lamda_entropy)

        self.Transition_initial = (
            torch.eye(self.d_dim, device=self.device) * (1 - 0.05 * self.d_dim)
            + torch.ones((self.d_dim, self.d_dim), device=self.device) * 0.05
        )

        # d prior transition matrix
        self.dprior = nn.Sequential(
            nn.Linear(d_dim, d_dim),
            nn.Softmax(dim=1)
        )

        # z transition / prior networks
        self.ztrainsition_list = nn.ModuleList()
        self.ztrainsition_mean_list = nn.ModuleList()
        self.ztrainsition_std_list = nn.ModuleList()

        # d posterior networks
        self.dposterior_list = nn.ModuleList()

        # z posterior networks
        self.zposterior_list = nn.ModuleList()
        self.zposterior_mean_list = nn.ModuleList()
        self.zposterior_std_list = nn.ModuleList()

        # y emission networks
        self.yemission_list = nn.ModuleList()
        self.yemission_mean_list = nn.ModuleList()
        self.yemission_std_list = nn.ModuleList()

        for _ in range(self.d_dim):

            dposterior = nn.Sequential(
                nn.Linear(h_dim, d_dim),
                nn.Softmax(dim=1)
            )
            self.dposterior_list.append(dposterior)

            zposterior = nn.Sequential(
                nn.Linear(z_dim + h_dim, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, z_dim),
                nn.ReLU()
            )
            zposterior_mean = nn.Linear(z_dim, z_dim)
            zposterior_std = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Softplus()
            )

            self.zposterior_list.append(zposterior)
            self.zposterior_mean_list.append(zposterior_mean)
            self.zposterior_std_list.append(zposterior_std)

            ztrainsition = nn.Sequential(
                nn.Linear(z_dim + h_dim, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, z_dim),
                nn.ReLU()
            )
            ztrainsition_mean = nn.Linear(z_dim, z_dim)
            ztrainsition_std = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Softplus()
            )

            self.ztrainsition_list.append(ztrainsition)
            self.ztrainsition_mean_list.append(ztrainsition_mean)
            self.ztrainsition_std_list.append(ztrainsition_std)

            yemission = nn.Sequential(
                nn.Linear(z_dim + h_dim, y_dim),
                nn.ReLU(),
                nn.Linear(y_dim, y_dim),
                nn.ReLU()
            )
            yemission_mean = nn.Linear(y_dim, y_dim)
            yemission_std = nn.Sequential(
                nn.Linear(y_dim, y_dim),
                nn.Softplus()
            )

            self.yemission_list.append(yemission)
            self.yemission_mean_list.append(yemission_mean)
            self.yemission_std_list.append(yemission_std)

        # Recurrence
        self.rnn_forward = nn.GRU(
            x_dim,
            h_dim,
            n_layers,
            bidirectional=False
        )

        if self.bidirection:
            self.rnn_backward = nn.GRU(
                y_dim + h_dim,
                int(h_dim / 2),
                n_layers,
                bidirectional=self.bidirection
            )
        else:
            self.rnn_backward = nn.GRU(
                y_dim + h_dim,
                h_dim,
                n_layers,
                bidirectional=self.bidirection
            )

    def TransitionMatrix(self):
        if self.dataname == "Sleep":
            Transition = (
                self.dprior(self.Transition_initial) * 0.2
                + torch.eye(self.d_dim, device=self.device) * 0.8
            )
        else:
            Transition = (
                self.dprior(self.Transition_initial) / 2
                + torch.eye(self.d_dim, device=self.device) / 2
            )

        return Transition

    def forward(self, x, y):
        """
        x shape: [time, batch, x_dim]
        y shape: [time, batch, y_dim]
        """

        Transition = self.TransitionMatrix()

        all_d_posterior = []       # posterior probability vectors
        all_d_t_sampled_plot = []  # sampled state labels
        all_d_t_sampled = []       # sampled one-hot vectors

        all_z_posterior_mean = []
        all_z_posterior_std = []
        all_z_t_sampled = []

        all_y_emission_mean = []
        all_y_emission_std = []

        kld_gaussian_loss = 0.0
        kld_category_loss = 0.0
        nll_loss = 0.0

        batch_size = x.size(1)

        h0 = torch.zeros(
            (self.n_layers, batch_size, self.h_dim),
            device=self.device
        )

        if self.bidirection:
            A0 = torch.zeros(
                (self.n_layers * 2, batch_size, int(self.h_dim / 2)),
                device=self.device
            )
        else:
            A0 = torch.zeros(
                (self.n_layers, batch_size, self.h_dim),
                device=self.device
            )

        # Initial discrete state d0
        # GPU-safe sampling
        initial_probs = torch.ones((self.d_dim,), device=self.device) / self.d_dim
        samples = torch.distributions.Categorical(initial_probs).sample((batch_size,))

        d0 = self._one_hot_encode(samples, self.d_dim)

        all_d_posterior.append(
            torch.ones((batch_size, self.d_dim), device=self.device) / self.d_dim
        )
        all_d_t_sampled_plot.append(samples.reshape(-1, 1))
        all_d_t_sampled.append(d0)

        # Initial continuous state z0
        z0 = torch.zeros((batch_size, self.z_dim), device=self.device)
        all_z_posterior_std.append(z0)
        all_z_posterior_mean.append(z0)
        all_z_t_sampled.append(z0)

        # Forward RNN
        output_forward, h_forward = self.rnn_forward(x, h0)

        # Backward RNN
        yh_concatenate = torch.cat([y, output_forward], dim=2)
        yh_concatenate_inverse = torch.flip(yh_concatenate, [0])

        output_backward, h_backward = self.rnn_backward(
            yh_concatenate_inverse,
            A0
        )

        for t in range(x.size(0)):

            # d prior
            d_prior = torch.mm(all_d_t_sampled[t], Transition)

            # d posterior
            d_posterior_list = []
            d_posterior = 0.0

            for i in range(self.d_dim):
                posterior_i = self.dposterior_list[i](
                    output_backward[x.size(0) - t - 1]
                )
                d_posterior_list.append(posterior_i)

                d_posterior = (
                    d_posterior
                    + posterior_i * all_d_t_sampled[t][:, i:(i + 1)]
                )

            all_d_posterior.append(d_posterior)

            # Sample discrete state d_t
            d_t_samples = torch.distributions.Categorical(d_posterior).sample()
            all_d_t_sampled_plot.append(d_t_samples.reshape(-1, 1))
            all_d_t_sampled.append(
                self._one_hot_encode(d_t_samples, self.d_dim)
            )

            # z prior and z posterior
            z_prior_list = []
            z_prior_mean_list = []
            z_prior_std_list = []

            z_posterior_list = []
            z_posterior_mean_list = []
            z_posterior_std_list = []

            z_posterior_mean = 0.0
            z_posterior_std = 0.0

            for i in range(self.d_dim):
                z_prior_i = self.ztrainsition_list[i](
                    torch.cat([output_forward[t], all_z_t_sampled[t]], dim=1)
                )
                z_prior_list.append(z_prior_i)

                z_prior_mean_i = self.ztrainsition_mean_list[i](z_prior_i)
                z_prior_std_i = self.ztrainsition_std_list[i](z_prior_i)

                z_prior_mean_list.append(z_prior_mean_i)
                z_prior_std_list.append(z_prior_std_i)

                z_posterior_i = self.zposterior_list[i](
                    torch.cat(
                        [
                            output_backward[x.size(0) - t - 1],
                            all_z_t_sampled[t]
                        ],
                        dim=1
                    )
                )
                z_posterior_list.append(z_posterior_i)

                z_posterior_mean_i = self.zposterior_mean_list[i](z_posterior_i)
                z_posterior_std_i = self.zposterior_std_list[i](z_posterior_i)

                z_posterior_mean_list.append(z_posterior_mean_i)
                z_posterior_std_list.append(z_posterior_std_i)

                z_posterior_mean = (
                    z_posterior_mean
                    + z_posterior_mean_i * all_d_t_sampled[t + 1][:, i:(i + 1)]
                )
                z_posterior_std = (
                    z_posterior_std
                    + z_posterior_std_i * all_d_t_sampled[t + 1][:, i:(i + 1)]
                )

            all_z_posterior_mean.append(z_posterior_mean)
            all_z_posterior_std.append(z_posterior_std)

            # Reparameterized continuous sample z_t
            z_t = self._reparameterized_normal_sample(
                z_posterior_mean,
                z_posterior_std
            )
            all_z_t_sampled.append(z_t)

            # y emission
            y_emission_list = []
            y_emission_mean_list = []
            y_emission_std_list = []

            for i in range(self.d_dim):
                y_emission_i = self.yemission_list[i](
                    torch.cat([output_forward[t], all_z_t_sampled[t + 1]], dim=1)
                )
                y_emission_list.append(y_emission_i)

                y_emission_mean_i = self.yemission_mean_list[i](y_emission_i)
                y_emission_std_i = self.yemission_std_list[i](y_emission_i)

                y_emission_mean_list.append(y_emission_mean_i)
                y_emission_std_list.append(y_emission_std_i)

            # Losses
            for i in range(self.d_dim):
                kld_gaussian_loss = kld_gaussian_loss + torch.sum(
                    self._kld_gauss(
                        z_posterior_mean_list[i],
                        z_posterior_std_list[i],
                        z_prior_mean_list[i],
                        z_prior_std_list[i]
                    )
                    * d_posterior[:, i:(i + 1)]
                )

            for i in range(self.d_dim):
                kld_category_loss = kld_category_loss + torch.sum(
                    self._kld_category(
                        d_posterior_list[i],
                        Transition[i:(i + 1), :]
                    )
                    * all_d_posterior[-2][:, i]
                )

            for i in range(self.d_dim):
                nll_loss = nll_loss + torch.sum(
                    self._nll_gauss(
                        y_emission_mean_list[i],
                        y_emission_std_list[i],
                        y[t]
                    )
                    * d_posterior[:, i:(i + 1)]
                )

            # Store selected emission by sampled state
            y_emission_mean = 0.0
            y_emission_std = 0.0

            for i in range(self.d_dim):
                y_emission_mean = (
                    y_emission_mean
                    + y_emission_mean_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]
                )
                y_emission_std = (
                    y_emission_std
                    + y_emission_std_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]
                )

            all_y_emission_mean.append(y_emission_mean)
            all_y_emission_std.append(y_emission_std)

        d_probs_tensor = torch.stack(all_d_posterior, dim=0)  # [T+1, B, K]
        balance_loss = self._regime_balance_loss(d_probs_tensor)
        entropy_loss = self._posterior_entropy_loss(d_probs_tensor)

        return (
            kld_gaussian_loss,
            kld_category_loss,
            nll_loss,
            balance_loss,
            entropy_loss,
            (all_z_posterior_mean, all_z_posterior_std),
            (all_y_emission_mean, all_y_emission_std),
            all_d_t_sampled_plot,
            all_z_t_sampled,
            all_d_posterior,
            all_d_t_sampled
        )

    def _regime_balance_loss(self, d_probs):
        """
        Balance penalty for regime identification.
        Encourages the average regime usage to be close to uniform across all regimes.
        """
        if d_probs.size(0) > 1:
            d_probs = d_probs[1:]
        usage = d_probs.mean(dim=(0, 1))
        target = torch.full_like(usage, 1.0 / self.d_dim)
        return torch.sum((usage - target) ** 2)

    def _posterior_entropy_loss(self, d_probs):
        """Optional entropy term. Positive lamda_entropy encourages sharper regime probabilities."""
        if d_probs.size(0) > 1:
            d_probs = d_probs[1:]
        eps = 1e-12
        d_probs = d_probs.clamp_min(eps)
        entropy = -torch.sum(d_probs * torch.log(d_probs), dim=-1).mean()
        return entropy

    def _forecastingMultiStep(self, x, y, step=1, S=1):
        with torch.no_grad():

            Transition = self.TransitionMatrix()

            h0 = torch.zeros(
                (self.n_layers, x.size(1), self.h_dim),
                device=self.device
            )

            output_forward, h_forward = self.rnn_forward(x, h0)

            forecast_MC = []
            forecast_d_MC = []
            forecast_z_MC = []

            for _ in range(S):

                (
                    kld_gaussian_loss,
                    kld_category_loss,
                    nll_loss,
                    balance_loss,
                    entropy_loss,
                    (all_z_posterior_mean, all_z_posterior_std),
                    (all_y_emission_mean, all_y_emission_std),
                    all_d_t_sampled_plot,
                    z_t_sampled,
                    all_d_posterior,
                    all_d_t_sampled
                ) = self.forward(x, y)

                forecast_x = []
                forecast_y = []

                forecast_x.append(y[-1, :, :].unsqueeze(0))

                for t in range(step):

                    _, h_forward = self.rnn_forward(forecast_x[t], h_forward)

                    # d prior
                    d_prior = torch.mm(all_d_t_sampled[-1], Transition)

                    # GPU-safe sample d from prior
                    samples = torch.distributions.Categorical(d_prior).sample()
                    d_t_sampled = self._one_hot_encode(samples, self.d_dim)

                    all_d_t_sampled_plot.append(samples.reshape(-1, 1))
                    all_d_t_sampled.append(d_t_sampled)
                    all_d_posterior.append(d_prior)

                    # z prior
                    z_prior_mean = 0.0
                    z_prior_std = 0.0

                    for i in range(self.d_dim):
                        z_prior_i = self.ztrainsition_list[i](
                            torch.cat(
                                [h_forward.squeeze(0), z_t_sampled[-1]],
                                dim=1
                            )
                        )

                        z_prior_mean_i = self.ztrainsition_mean_list[i](z_prior_i)
                        z_prior_std_i = self.ztrainsition_std_list[i](z_prior_i)

                        z_prior_mean = (
                            z_prior_mean
                            + z_prior_mean_i * d_t_sampled[:, i:(i + 1)]
                        )
                        z_prior_std = (
                            z_prior_std
                            + z_prior_std_i * d_t_sampled[:, i:(i + 1)]
                        )

                    z_prior_std = z_prior_std.clamp_min(1e-6)

                    # sample z
                    z_t = torch.distributions.Normal(
                        z_prior_mean,
                        z_prior_std
                    ).sample()

                    z_t_sampled.append(z_t)

                    all_z_posterior_mean.append(z_prior_mean)
                    all_z_posterior_std.append(z_prior_std)

                    # y emission
                    y_emission_mean = 0.0
                    y_emission_std = 0.0

                    for i in range(self.d_dim):
                        y_emission_i = self.yemission_list[i](
                            torch.cat(
                                [h_forward.squeeze(0), z_t_sampled[-1]],
                                dim=1
                            )
                        )

                        y_emission_mean_i = self.yemission_mean_list[i](y_emission_i)
                        y_emission_std_i = self.yemission_std_list[i](y_emission_i)

                        y_emission_mean = (
                            y_emission_mean
                            + y_emission_mean_i * d_t_sampled[:, i:(i + 1)]
                        )
                        y_emission_std = (
                            y_emission_std
                            + y_emission_std_i * d_t_sampled[:, i:(i + 1)]
                        )

                    y_emission_std = y_emission_std.clamp_min(1e-6)

                    all_y_emission_mean.append(y_emission_mean)
                    all_y_emission_std.append(y_emission_std)

                    y_t = torch.distributions.Normal(
                        y_emission_mean,
                        y_emission_std
                    ).sample().unsqueeze(0)

                    forecast_x.append(y_t)
                    forecast_y.append(y_t.squeeze(0).cpu().numpy())

                forecast_MC.append(forecast_y)
                forecast_d_MC.append(all_d_t_sampled_plot)
                forecast_z_MC.append(z_t_sampled)

            forecast_MC = np.array(forecast_MC)

            forecast_z_MC = torch.stack(
                [
                    torch.stack(forecast_z_MC[i])
                    for i in range(len(forecast_z_MC))
                ]
            ).cpu().numpy()

            forecast_d_MC = torch.stack(
                [
                    torch.stack(forecast_d_MC[i])
                    for i in range(len(forecast_d_MC))
                ]
            ).cpu().numpy()

        return forecast_MC, forecast_d_MC, forecast_z_MC

    def _reparameterized_normal_sample(self, mean, std):
        """Reparameterized Gaussian sampling."""
        std = std.clamp_min(1e-6)
        eps = torch.randn_like(std, device=self.device)
        return eps.mul(std).add_(mean)

    def _reparameterized_category_gumbel_softmax_sample(self, logits):
        """Gumbel-softmax categorical sample."""
        if self.temperature > 0.01:
            self.temperature = self.temperature / 1.001
        else:
            self.temperature = 0.01

        logits = logits.clamp_min(1e-12)

        gumbel = torch.distributions.Gumbel(
            torch.tensor([0.0], device=self.device),
            torch.tensor([1.0], device=self.device)
        ).sample(logits.size()).squeeze()

        y = torch.log(logits) + gumbel

        return torch.nn.functional.softmax((y / self.temperature), dim=1)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        eps = 1e-6
        std_1 = std_1.clamp_min(eps)
        std_2 = std_2.clamp_min(eps)

        kld_element = (
            2 * torch.log(std_2)
            - 2 * torch.log(std_1)
            + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2)
            - 1
        )

        return 0.5 * kld_element

    def _kld_category(self, d_posterior, d_prior):
        """Categorical KL divergence."""
        eps = 1e-12

        d_posterior = d_posterior.clamp_min(eps)
        d_prior = d_prior.clamp_min(eps)

        return torch.sum(
            torch.mul(
                torch.log(torch.div(d_posterior, d_prior)),
                d_posterior
            ),
            dim=1
        )

    def _nll_bernoulli(self, theta, x):
        #negative log-likelihood
        eps = 1e-12
        theta = theta.clamp(eps, 1 - eps)

        return -torch.sum(
            x * torch.log(theta)
            + (1 - x) * torch.log(1 - theta)
        )

    def _nll_gauss(self, mean, std, x):
        #Gaussian negative log-likelihood
        eps = 1e-6
        std = std.clamp_min(eps)

        return (
            0.5 * torch.log(torch.tensor(2 * math.pi, device=self.device))
            + torch.log(std)
            + (x - mean).pow(2) / (2 * std.pow(2))
        )

    def _one_hot_encode(self, x, n_classes):
        """One-hot encode integer labels."""
        x = x.to(self.device).long()
        return torch.eye(n_classes, device=self.device)[x]


def train(model, optimizer, trainX, trainY, epoch, batch_size, n_epochs, status="train"):

    model.train()

    if epoch < n_epochs / 2:
        annealing = 0.01
    else:
        annealing = min(1.0, 0.01 + epoch / n_epochs / 2)

    print("Annealing coef:", annealing)

    for batch in range(0, trainX.size(1), batch_size):

        batchX = trainX[:, batch:(batch + batch_size), :]
        batchY = trainY[:, batch:(batch + batch_size), :]

        (
            kld_gaussian_loss,
            kld_category_loss,
            nll_loss,
            balance_loss,
            entropy_loss,
            *_
        ) = model(batchX, batchY)

        kld_loss = kld_gaussian_loss + kld_category_loss

        normalizer = batchX.size(1) * batchX.size(0)

        loss = (
            annealing * kld_loss / normalizer
            + nll_loss / normalizer
            + model.lamda_b * balance_loss
            + model.lamda_entropy * entropy_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    (
        all_d_t_sampled_plot,
        all_z_t_sampled,
        loss,
        all_d_posterior,
        all_z_posterior_mean
    ) = test(model, trainX, trainY, epoch, "train")

    return (
        all_d_t_sampled_plot,
        all_z_t_sampled,
        loss,
        all_d_posterior,
        all_z_posterior_mean
    )


def test(model, testX, testY, epoch, status="test"):
    #Evaluate likelihood of the model
    model.eval()

    with torch.no_grad():
        size = testX.size(1) * testX.size(0)

        (
            kld_gaussian_loss,
            kld_category_loss,
            nll_loss,
            balance_loss,
            entropy_loss,
            (all_z_posterior_mean, all_z_posterior_std),
            (all_y_emission_mean, all_y_emission_std),
            all_d_t_sampled_plot,
            all_z_t_sampled,
            all_d_posterior,
            all_d_t_sampled
        ) = model(testX, testY)

        nll_loss_total = nll_loss.item()
        kld_gaussian_loss_total = kld_gaussian_loss.item()
        kld_category_loss_total = kld_category_loss.item()
        balance_loss_total = balance_loss.item()
        entropy_loss_total = entropy_loss.item()

        loss = (
            kld_gaussian_loss_total
            + kld_category_loss_total
            + nll_loss_total
            + model.lamda_b * balance_loss_total * size
            + model.lamda_entropy * entropy_loss_total * size
        )

        print(
            "{} Epoch:{}\t KLD_Gaussian Loss: {:.6f}, "
            "KLD_Category Loss: {:.6f}, NLL Loss: {:.6f}, "
            "Balance Loss: {:.6f}, Entropy: {:.6f}, Loss: {:.4f}".format(
                status,
                epoch,
                kld_gaussian_loss_total / size,
                kld_category_loss_total / size,
                nll_loss_total / size,
                balance_loss_total,
                entropy_loss_total,
                loss / size
            )
        )

        all_d_t_sampled = torch.stack(
            all_d_t_sampled
        ).cpu().numpy().transpose((1, 0, 2))

        all_d_t_sampled_plot = torch.stack(
            all_d_t_sampled_plot
        ).cpu().numpy().transpose((1, 0, 2))

        all_z_t_sampled = torch.stack(
            all_z_t_sampled
        ).cpu().numpy().transpose((1, 0, 2))

        all_d_posterior = torch.stack(
            all_d_posterior
        ).cpu().numpy().transpose((1, 0, 2))

        all_z_posterior_mean = torch.stack(
            all_z_posterior_mean
        ).cpu().numpy().transpose((1, 0, 2))

    return (
        all_d_t_sampled_plot,
        all_z_t_sampled,
        loss / size,
        all_d_posterior,
        all_z_posterior_mean
    )



def train_dsssm_model(
    model,
    optimizer,
    trainX,
    trainY,
    validX,
    validY,
    n_epochs=40,
    batch_size=64,
    patience=8,
    verbose=True,
):
    """
    CUDA-friendly training wrapper for the market-regime DSSSM.

    The model is trained with the original DS3M/DSSSM objective plus optional
    regime-identification regularizers already defined in the model:

        loss = NLL + annealing * (Gaussian_KL + Categorical_KL)
               + lamda_b * regime_balance_loss
               + lamda_entropy * posterior_entropy_loss

    Returns a history dictionary and reloads the best validation model weights.
    """
    import copy as _copy
    import numpy as _np

    history = {
        "train_loss": [], "val_loss": [],
        "train_nll": [], "val_nll": [],
        "train_kld_gauss": [], "val_kld_gauss": [],
        "train_kld_cat": [], "val_kld_cat": [],
        "train_balance": [], "val_balance": [],
        "train_entropy": [], "val_entropy": [],
    }

    best_val_loss = _np.inf
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()

        if epoch < n_epochs / 2:
            annealing = 0.01
        else:
            annealing = min(1.0, 0.01 + epoch / n_epochs / 2)

        accum = {
            "loss": 0.0, "nll": 0.0, "kld_g": 0.0, "kld_c": 0.0,
            "balance": 0.0, "entropy": 0.0,
        }
        n_batches = 0

        for batch in range(0, trainX.size(1), batch_size):
            batchX = trainX[:, batch:(batch + batch_size), :]
            batchY = trainY[:, batch:(batch + batch_size), :]

            (
                kld_gaussian_loss,
                kld_category_loss,
                nll_loss,
                balance_loss,
                entropy_loss,
                *_
            ) = model(batchX, batchY)

            normalizer = batchX.size(0) * batchX.size(1)
            loss = (
                nll_loss / normalizer
                + annealing * (kld_gaussian_loss + kld_category_loss) / normalizer
                + model.lamda_b * balance_loss
                + model.lamda_entropy * entropy_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            accum["loss"] += float(loss.detach().cpu())
            accum["nll"] += float((nll_loss / normalizer).detach().cpu())
            accum["kld_g"] += float((kld_gaussian_loss / normalizer).detach().cpu())
            accum["kld_c"] += float((kld_category_loss / normalizer).detach().cpu())
            accum["balance"] += float(balance_loss.detach().cpu())
            accum["entropy"] += float(entropy_loss.detach().cpu())
            n_batches += 1

        for k in accum:
            accum[k] /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            (
                kld_g_val,
                kld_c_val,
                nll_val,
                balance_val,
                entropy_val,
                *_
            ) = model(validX, validY)

            normalizer_val = validX.size(0) * validX.size(1)
            val_loss = (
                nll_val / normalizer_val
                + (kld_g_val + kld_c_val) / normalizer_val
                + model.lamda_b * balance_val
                + model.lamda_entropy * entropy_val
            )

            val_loss_float = float(val_loss.detach().cpu())
            val_nll_float = float((nll_val / normalizer_val).detach().cpu())
            val_kld_g_float = float((kld_g_val / normalizer_val).detach().cpu())
            val_kld_c_float = float((kld_c_val / normalizer_val).detach().cpu())
            val_balance_float = float(balance_val.detach().cpu())
            val_entropy_float = float(entropy_val.detach().cpu())

        history["train_loss"].append(accum["loss"])
        history["val_loss"].append(val_loss_float)
        history["train_nll"].append(accum["nll"])
        history["val_nll"].append(val_nll_float)
        history["train_kld_gauss"].append(accum["kld_g"])
        history["val_kld_gauss"].append(val_kld_g_float)
        history["train_kld_cat"].append(accum["kld_c"])
        history["val_kld_cat"].append(val_kld_c_float)
        history["train_balance"].append(accum["balance"])
        history["val_balance"].append(val_balance_float)
        history["train_entropy"].append(accum["entropy"])
        history["val_entropy"].append(val_entropy_float)

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={accum['loss']:.6f} | "
                f"val_loss={val_loss_float:.6f} | "
                f"val_nll={val_nll_float:.6f} | "
                f"val_kld_cat={val_kld_c_float:.6f} | "
                f"val_balance={val_balance_float:.6f} | "
                f"annealing={annealing:.4f}"
            )

        if val_loss_float < best_val_loss:
            best_val_loss = val_loss_float
            best_state_dict = _copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history

# Regime-identification helpers

def extract_dsssm_regimes(model, X, Y, use_posterior_argmax=True):
    model.eval()
    with torch.no_grad():
        (
            kld_gaussian_loss,
            kld_category_loss,
            nll_loss,
            balance_loss,
            entropy_loss,
            _,
            _,
            all_d_t_sampled_plot,
            _,
            all_d_posterior,
            all_d_t_sampled,
        ) = model(X, Y)

        probs_tbk = torch.stack(all_d_posterior, dim=0)[1:]  # remove initial uniform state
        probs = probs_tbk.permute(1, 0, 2).detach().cpu().numpy()  # [B, T, K]

        if use_posterior_argmax:
            states = np.argmax(probs, axis=-1)
        else:
            sampled = torch.stack(all_d_t_sampled_plot, dim=0)[1:]
            states = sampled.permute(1, 0, 2).squeeze(-1).detach().cpu().numpy().astype(int)

    return states.astype(int), probs


def transition_matrix_from_states(states, n_states=None):
    arr = np.asarray(states, dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if n_states is None:
        n_states = int(np.nanmax(arr)) + 1

    counts = np.zeros((n_states, n_states), dtype=float)
    for seq in arr:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[int(a), int(b)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)


def effective_number_of_regimes(usage):
    p = np.asarray(usage, dtype=float)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def regime_diagnostics(states, n_states=None):
    arr = np.asarray(states, dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if n_states is None:
        n_states = int(arr.max()) + 1

    flat = arr.reshape(-1)
    counts = np.bincount(flat, minlength=n_states).astype(float)
    usage = counts / counts.sum()
    switch_rate = float(np.mean(arr[:, 1:] != arr[:, :-1])) if arr.shape[1] > 1 else np.nan

    return {
        "usage": usage,
        "min_usage": float(usage.min()),
        "max_usage": float(usage.max()),
        "effective_regimes": effective_number_of_regimes(usage),
        "switch_rate": switch_rate,
        "transition_matrix": transition_matrix_from_states(arr, n_states),
    }


def daily_segment_summary(states, dates=None):
    arr = np.asarray(states, dtype=int)
    if arr.ndim != 2:
        raise ValueError("states must have shape [n_days, T] for daily segment diagnostics")
    if dates is None:
        dates = np.arange(arr.shape[0])

    rows = []
    for i, seq in enumerate(arr):
        switches = int(np.sum(seq[1:] != seq[:-1]))
        rows.append({"date": dates[i], "switches": switches, "segments": switches + 1})

    df = pd.DataFrame(rows)
    summary = {
        "median_segments": float(df["segments"].median()),
        "mean_segments": float(df["segments"].mean()),
        "p75_segments": float(df["segments"].quantile(0.75)),
        "p90_segments": float(df["segments"].quantile(0.90)),
        "max_segments": int(df["segments"].max()),
        "one_state_share": float((df["segments"] == 1).mean()),
        "multi_state_share": float((df["segments"] > 1).mean()),
    }
    return summary, df


def align_market_panel_with_states(panel, states, state_col="state_dsssm"):
    out = panel.copy()
    flat_states = np.asarray(states, dtype=int).reshape(-1)
    if len(out) != len(flat_states):
        raise ValueError(
            f"Panel length ({len(out)}) does not match flattened states ({len(flat_states)}). "
            "Align the panel by dropping the same time steps used to create X/Y."
        )
    out[state_col] = flat_states
    return out


def state_economic_summary(panel, state_col, econ_cols):
    return panel.groupby(state_col)[econ_cols].mean()


def future_return_summary(panel, state_col, future_cols=("future_ret_30", "future_ret_60", "future_ret_120")):
    cols = [c for c in future_cols if c in panel.columns]
    if not cols:
        raise ValueError("None of the requested future-return columns are in the panel.")
    return panel.groupby(state_col)[cols].agg(["mean", "std", "count"])


def regime_aware_return(panel, state_col, return_col="ret_1", date_col="date", invest_state=None):
    if invest_state is None:
        raise ValueError("invest_state must be chosen on validation data before evaluating test data.")
    tmp = panel.copy()
    tmp["invest"] = (tmp[state_col] == invest_state).astype(float)
    tmp["strategy_ret"] = tmp["invest"] * tmp[return_col]
    daily = tmp.groupby(date_col)["strategy_ret"].sum()
    return {
        "invest_state": int(invest_state),
        "avg_daily_regime_return": float(daily.mean()),
        "daily_regime_returns": daily,
    }


def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

    plt.bar(
        np.arange(len(max_grads)),
        [g.cpu().numpy() for g in max_grads],
        alpha=0.2,
        lw=1,
        color="c"
    )

    plt.bar(
        np.arange(len(ave_grads)),
        [g.cpu().numpy() for g in ave_grads],
        alpha=0.2,
        lw=1,
        color="b"
    )

    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4)
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"]
    )

    plt.tight_layout()
    plt.show()


class EarlyStopping:

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1

            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""

        if self.verbose:
            self.trace_func(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                f"Saving model ..."
            )

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss