import numpy as np
import torch
import torch.optim as optim
from mixture_vae import MixtureVAE


class VAEModule:
    """
    Mixture-VAE wrapper for unsupervised training on sequence data.
    Input expected shape: [batch, seq_len, feature].
    """
    def __init__(self, model_params):
        if model_params.name != "mixture_vae":
            raise ValueError("Only mixture_vae is supported in this module.")

        self.model = MixtureVAE(model_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(
        self,
        train_loader,
        val_loader=None,
        lr=1e-3,
        epochs=200,
        weight_decay=0.0,
        patience=20,
        verbose=True,
    ):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = np.inf
        best_state = None
        wait = 0

        history = {
            "train_loss": [],
            "val_loss": [],

            # Unweighted validation loss components.
            "val_loss_r": [],
            "val_loss_m": [],
            "val_loss_i": [],
            "val_loss_t": [],
            "val_loss_b": [],

            # Regime diagnostics.
            "val_regime_usage": [],
            "val_effective_regimes": [],
            "val_switch_rate": [],
        }

        for epoch in range(epochs):
            # ---- train ----
            self.model.train()
            total_train_loss = 0.0
            total_train_n = 0

            for batch in train_loader:
                x_batch = batch[0].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                _ = self.model(x_batch)
                loss = self.model.loss
                loss.backward()
                optimizer.step()

                batch_size = x_batch.size(0)
                total_train_loss += loss.item()
                total_train_n += batch_size

            avg_train_loss = total_train_loss / max(total_train_n, 1)
            history["train_loss"].append(avg_train_loss)

            # ---- validation ----
            avg_val_loss = None
            usage = None
            effective_regimes = None
            switch_rate = None

            if val_loader is not None:
                self.model.eval()

                total_val_loss = 0.0
                total_val_n = 0

                comp_sums = {
                    "loss_r": 0.0,
                    "loss_m": 0.0,
                    "loss_i": 0.0,
                    "loss_t": 0.0,
                    "loss_b": 0.0,
                }

                val_probs = []

                with torch.no_grad():
                    for batch in val_loader:
                        x_batch = batch[0].to(self.device, non_blocking=True)

                        _ = self.model(x_batch)
                        loss = self.model.loss
                        comps = self.model.loss_components

                        batch_size = x_batch.size(0)
                        total_val_loss += loss.item()
                        total_val_n += batch_size

                        for k in comp_sums:
                            comp_sums[k] += comps[k].item()

                        s_prob = self.model.get_s_prob(x_batch)  # [B, T, K]
                        val_probs.append(s_prob.detach().cpu())

                avg_val_loss = total_val_loss / max(total_val_n, 1)
                history["val_loss"].append(avg_val_loss)

                avg_comps = {k: v / max(total_val_n, 1) for k, v in comp_sums.items()}
                history["val_loss_r"].append(avg_comps["loss_r"])
                history["val_loss_m"].append(avg_comps["loss_m"])
                history["val_loss_i"].append(avg_comps["loss_i"])
                history["val_loss_t"].append(avg_comps["loss_t"])
                history["val_loss_b"].append(avg_comps["loss_b"])

                if len(val_probs) > 0:
                    val_probs_tensor = torch.cat(val_probs, dim=0)  # [N, T, K]

                    usage = val_probs_tensor.mean(dim=(0, 1)).numpy()
                    effective_regimes = 1.0 / np.sum(usage ** 2)

                    val_states = torch.argmax(val_probs_tensor, dim=-1).numpy()
                    switch_rate = np.mean(val_states[:, 1:] != val_states[:, :-1])

                    history["val_regime_usage"].append(usage)
                    history["val_effective_regimes"].append(float(effective_regimes))
                    history["val_switch_rate"].append(float(switch_rate))

                # Early stopping uses total validation objective within this run only.
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch+1}. "
                                f"Best val loss = {best_val_loss:.6f}"
                            )
                        break

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                if avg_val_loss is not None:
                    msg = (
                        f"Epoch [{epoch+1}/{epochs}] | "
                        f"train_loss={avg_train_loss:.6f} | "
                        f"val_total={avg_val_loss:.6f} | "
                        f"val_recon={history['val_loss_r'][-1]:.6f}"
                    )
                    if usage is not None:
                        msg += (
                            f" | usage={np.round(usage, 4)}"
                            f" | eff={effective_regimes:.3f}"
                            f" | switch={switch_rate:.4f}"
                        )
                    print(msg)
                else:
                    print(
                        f"Epoch [{epoch+1}/{epochs}] | "
                        f"train_loss={avg_train_loss:.6f}"
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def get_state_probabilities(self, data_loader):
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in data_loader:
                x_batch = batch[0].to(self.device, non_blocking=True)
                s_prob = self.model.get_s_prob(x_batch)  # [B, T, K]
                all_probs.append(s_prob.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def get_predicted_states(self, data_loader):
        probs = self.get_state_probabilities(data_loader)
        states = np.argmax(probs, axis=-1)  # [N, T]
        return states

    def get_embedding(self, data_loader):
        self.model.eval()
        all_x = []
        all_z = []

        with torch.no_grad():
            for batch in data_loader:
                x_batch = batch[0].to(self.device, non_blocking=True)
                mu_z, logsigma2_z = self.model.get_z(x_batch)
                all_x.append(x_batch.cpu().numpy())
                all_z.append(mu_z.cpu().numpy())

        all_x = np.concatenate(all_x, axis=0)
        all_z = np.concatenate(all_z, axis=0)
        return all_x, all_z
