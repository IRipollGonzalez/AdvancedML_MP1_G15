
# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Extended for Week 1 programming exercises (02460):
# - Test ELBO evaluation
# - Aggregate posterior plotting (PCA if latent_dim > 2)
# - Mixture-of-Gaussians (MoG) prior (KL via Monte Carlo)
# - Gaussian decoder for continuous MNIST
#
# Notes:
# - For non-Gaussian priors (e.g., MoG), KL(q||p) is estimated as E_q[log q - log p].

from __future__ import annotations

import argparse
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_torch_device(device_str: str) -> torch.device:
    """Convert a device string to a torch.device."""
    return torch.device(device_str)


@torch.no_grad()
def evaluate_average_elbo(
    model: "VAE",
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 1,
) -> float:
    """Evaluate average ELBO per datapoint on a dataset (e.g., test set)."""
    model.eval()
    total_elbo = 0.0
    total_count = 0

    for batch in data_loader:
        x = batch[0].to(device)
        elbo = model.elbo(x, n_samples=n_samples)  # scalar
        batch_size = x.shape[0]
        total_elbo += float(elbo.item()) * batch_size
        total_count += batch_size

    return total_elbo / max(total_count, 1)


@torch.no_grad()
def sample_from_aggregate_posterior(
    model: "VAE",
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_points: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ONE latent point z ~ q(z|x) per datapoint from the dataset."""
    model.eval()
    zs = []
    ys = []
    count = 0

    for x, y in data_loader:
        x = x.to(device)
        q = model.encoder(x)
        z = q.rsample()  # (B, M)
        zs.append(z.cpu())
        ys.append(y.cpu())
        count += x.shape[0]
        if count >= max_points:
            break

    z_samples = torch.cat(zs, dim=0)[:max_points]
    labels = torch.cat(ys, dim=0)[:max_points]
    return z_samples, labels


def project_to_2d(z: torch.Tensor) -> torch.Tensor:
    """Project latent samples to 2D (identity if dim=2, otherwise PCA)."""
    if z.shape[1] == 2:
        return z

    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for PCA when latent_dim > 2. "
            "Install it with: pip install scikit-learn"
        ) from e

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z.detach().cpu().numpy())
    return torch.from_numpy(z_2d).float()


@torch.no_grad()
def plot_aggregate_posterior(
    model: "VAE",
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: str,
    max_points: int = 10_000,
) -> None:
    """Plot aggregate posterior samples and color by MNIST label."""
    import matplotlib.pyplot as plt

    z, y = sample_from_aggregate_posterior(model, data_loader, device, max_points=max_points)
    z2 = project_to_2d(z)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(z2[:, 0], z2[:, 1], c=y, s=6, alpha=0.7)
    plt.colorbar(sc, label="MNIST class")
    plt.title("Aggregate posterior samples (projected to 2D)")
    plt.xlabel("z[0] / PC1")
    plt.ylabel("z[1] / PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Priors
# ---------------------------

class GaussianPrior(nn.Module):
    """Standard Gaussian prior p(z) = N(0, I)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))

    def forward(self) -> td.Distribution:
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MoGPrior(nn.Module):
    """Mixture-of-Gaussians prior p(z) = sum_k pi_k N(z|mu_k, diag(sigma_k^2))."""

    def __init__(self, latent_dim: int, n_components: int = 10, trainable: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components

        logits = torch.zeros(n_components)
        means = torch.randn(n_components, latent_dim) * 0.05
        log_stds = torch.zeros(n_components, latent_dim)

        if trainable:
            self.logits = nn.Parameter(logits)
            self.means = nn.Parameter(means)
            self.log_stds = nn.Parameter(log_stds)
        else:
            self.register_buffer("logits", logits)
            self.register_buffer("means", means)
            self.register_buffer("log_stds", log_stds)

    def forward(self) -> td.Distribution:
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(loc=self.means, scale=torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(mix, comp)


# ---------------------------
# Encoders / Decoders
# ---------------------------

class GaussianEncoder(nn.Module):
    """Gaussian approximate posterior q_phi(z|x)."""

    def __init__(self, encoder_net: nn.Module):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x: torch.Tensor) -> td.Distribution:
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        std = torch.exp(log_std)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)


class BernoulliDecoder(nn.Module):
    """Bernoulli likelihood p(x|z) for binarized MNIST."""

    def __init__(self, decoder_net: nn.Module):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z: torch.Tensor) -> td.Distribution:
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GaussianDecoder(nn.Module):
    """Gaussian likelihood p(x|z) for continuous MNIST."""

    def __init__(
        self,
        decoder_net: nn.Module,
        fixed_sigma: Optional[float] = None,
        learn_per_pixel_sigma: bool = False,
        learn_global_sigma: bool = False,
    ):
        super().__init__()
        self.decoder_net = decoder_net

        active = sum([
            fixed_sigma is not None,
            learn_per_pixel_sigma,
            learn_global_sigma,
        ])
        if active != 1:
            raise ValueError(
                "Choose exactly one: fixed_sigma OR learn_per_pixel_sigma OR learn_global_sigma"
            )

        self.fixed_sigma = fixed_sigma
        self.learn_per_pixel_sigma = learn_per_pixel_sigma
        self.learn_global_sigma = learn_global_sigma

        if learn_per_pixel_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(28, 28))
        elif learn_global_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("_dummy", torch.zeros(1))

    def _sigma(self) -> torch.Tensor:
        if self.fixed_sigma is not None:
            return torch.tensor(self.fixed_sigma, device=next(self.parameters()).device)
        if self.learn_per_pixel_sigma:
            return torch.exp(self.log_sigma)
        if self.learn_global_sigma:
            return torch.exp(self.log_sigma)
        raise RuntimeError("Invalid sigma configuration")

    def forward(self, z: torch.Tensor) -> td.Distribution:
        mean = self.decoder_net(z)  # (B, 28, 28)
        sigma = self._sigma()

        if sigma.ndim == 0:
            scale = sigma * torch.ones_like(mean)
        elif sigma.shape == (1,):
            scale = sigma.view(1, 1, 1) * torch.ones_like(mean)
        else:
            scale = sigma.view(1, 28, 28) * torch.ones_like(mean)

        return td.Independent(td.Normal(loc=mean, scale=scale), 2)


# ---------------------------
# VAE
# ---------------------------

class VAE(nn.Module):
    """Variational Autoencoder with Monte Carlo ELBO."""

    def __init__(self, prior: nn.Module, decoder: nn.Module, encoder: nn.Module):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Monte Carlo estimate of ELBO, averaged over the batch."""
        q = self.encoder(x)
        pz = self.prior()

        z = q.rsample((n_samples,))  # (S, B, M)
        z_flat = z.view(-1, z.shape[-1])
        x_rep = x.repeat((n_samples,) + (1,) * (x.ndim - 1))

        log_px_given_z = self.decoder(z_flat).log_prob(x_rep).view(n_samples, -1)
        log_pz = pz.log_prob(z_flat).view(n_samples, -1)
        log_qz = q.log_prob(z_flat).view(n_samples, -1)

        elbo_per_x = (log_px_given_z + log_pz - log_qz).mean(dim=0)
        return elbo_per_x.mean()

    def forward(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        return -self.elbo(x, n_samples=n_samples)

    @torch.no_grad()
    def sample(self, n_samples: int = 1, return_mean: bool = False) -> torch.Tensor:
        """Sample x ~ p(x) via z ~ p(z) and x ~ p(x|z)."""
        self.eval()
        device = next(self.parameters()).device
        z = self.prior().sample((n_samples,)).to(device)
        px = self.decoder(z)
        if return_mean and hasattr(px, "mean"):
            return px.mean
        return px.sample()


# ---------------------------
# Training
# ---------------------------

def train(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    n_samples: int = 1,
) -> None:
    """Train a VAE by minimizing negative ELBO."""
    model.train()

    for epoch in range(epochs):
        epoch_bar = tqdm(data_loader, desc=f"Training epoch {epoch+1}/{epochs}")
        for batch in epoch_bar:
            x = batch[0].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(x, n_samples=n_samples)
            loss.backward()
            optimizer.step()

            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")


# ---------------------------
# Data / Nets
# ---------------------------

def build_mnist_loaders(
    batch_size: int,
    binarized: bool,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build MNIST dataloaders with optional binarization."""
    from torchvision import datasets, transforms

    if binarized:
        threshold = 0.5
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (threshold < x).float().squeeze()),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0]),
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def build_default_mlp(latent_dim: int) -> Tuple[nn.Module, nn.Module]:
    """Default MLP encoder/decoder (as in starter code)."""

    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, latent_dim * 2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    return encoder_net, decoder_net


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "sample", "eval", "plot_posterior"])
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--samples", type=str, default="samples.png")
    parser.add_argument("--posterior-plot", type=str, default="aggregate_posterior.png")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--elbo-samples", type=int, default=1)

    # Data / likelihood
    parser.add_argument("--binarized", action="store_true", help="use binarized MNIST (recommended for Bernoulli)")
    parser.add_argument("--decoder", type=str, default="bernoulli", choices=["bernoulli", "gaussian"])

    # Gaussian decoder sigma configuration
    parser.add_argument("--fixed-sigma", type=float, default=None)
    parser.add_argument("--learn-per-pixel-sigma", action="store_true")
    parser.add_argument("--learn-global-sigma", action="store_true")

    # Prior
    parser.add_argument("--prior", type=str, default="gaussian", choices=["gaussian", "mog"])
    parser.add_argument("--mog-components", type=int, default=10)

    # Posterior plot
    parser.add_argument("--max-posterior-points", type=int, default=10_000)

    args = parser.parse_args()

    print("# Options")
    for k, v in sorted(vars(args).items()):
        print(f"{k} = {v}")

    set_seed(args.seed)
    device = get_torch_device(args.device)

    # If decoder is Bernoulli, binarized MNIST is the standard choice.
    # If decoder is Gaussian, we typically use continuous MNIST.
    binarized = args.binarized if args.decoder == "bernoulli" else False
    train_loader, test_loader = build_mnist_loaders(args.batch_size, binarized=binarized)

    # Prior
    if args.prior == "gaussian":
        prior = GaussianPrior(args.latent_dim)
    else:
        prior = MoGPrior(args.latent_dim, n_components=args.mog_components, trainable=True)

    # Networks
    encoder_net, decoder_net = build_default_mlp(args.latent_dim)
    encoder = GaussianEncoder(encoder_net)

    if args.decoder == "bernoulli":
        decoder = BernoulliDecoder(decoder_net)
    else:
        decoder = GaussianDecoder(
            decoder_net=decoder_net,
            fixed_sigma=args.fixed_sigma,
            learn_per_pixel_sigma=args.learn_per_pixel_sigma,
            learn_global_sigma=args.learn_global_sigma,
        )

    model = VAE(prior=prior, decoder=decoder, encoder=encoder).to(device)

    if args.mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader, epochs=args.epochs, device=device, n_samples=args.elbo_samples)
        torch.save(model.state_dict(), args.model)
        print(f"Saved model to: {args.model}")

        test_elbo = evaluate_average_elbo(model, test_loader, device, n_samples=args.elbo_samples)
        print(f"Test ELBO (avg per datapoint): {test_elbo:.4f}")

    elif args.mode == "eval":
        model.load_state_dict(torch.load(args.model, map_location=device))
        test_elbo = evaluate_average_elbo(model, test_loader, device, n_samples=args.elbo_samples)
        print(f"Test ELBO (avg per datapoint): {test_elbo:.4f}")

    elif args.mode == "plot_posterior":
        model.load_state_dict(torch.load(args.model, map_location=device))
        plot_aggregate_posterior(
            model=model,
            data_loader=test_loader,
            device=device,
            out_path=args.posterior_plot,
            max_points=args.max_posterior_points,
        )
        print(f"Saved aggregate posterior plot to: {args.posterior_plot}")

    elif args.mode == "sample":
        from torchvision.utils import save_image

        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        with torch.no_grad():
            return_mean = (args.decoder == "gaussian")
            samples = model.sample(n_samples=64, return_mean=return_mean).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)
            print(f"Saved samples to: {args.samples}")


if __name__ == "__main__":
    main()
