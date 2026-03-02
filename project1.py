"""02460 Mini-project 1 runner (reproducible experiments).

This script is intentionally the ONLY new entrypoint needed for the mini-project.
It reuses the code you already completed in:
  - L1 - Deep latent variable models/vae_bernoulli.py
  - L2 - Flows/flow.py
  - L3 - Diffusion models/ddpm.py + unet.py
and the provided FID code in:
  - L4 - Mini project 1/fid.py

Important:
  - Do NOT modify fid.py unless the course staff explicitly tells you to.
  - This file orchestrates training/evaluation, saves plots/samples, and computes metrics.

Deliverables required by the project statement:
  Part A: VAE with Gaussian / MoG / Flow priors on binarized MNIST (Bernoulli likelihood)
  Part B: DDPM on MNIST and latent DDPM using a beta-VAE (Gaussian likelihood)
  Part C: Code snippets in the report (you will copy from the relevant functions).

Best practices baked in:
  - Deterministic seeding
  - Fixed output folder structure
  - Explicit device handling
  - Minimal, readable logging

Run:
  python project1.py --help
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Path setup: make LecturesCodes modules importable from this script location.
# -----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
LECTURES_DIR = THIS_DIR.parent

CODES_DIR = LECTURES_DIR / "LecturesCodes"

# Add LecturesCodes (and current dir) to sys.path
for p in [str(CODES_DIR), str(THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports after sys.path modifications
import vae_bernoulli  # type: ignore
import flow  # type: ignore
import ddpm  # type: ignore
import unet  # type: ignore
import fid  # type: ignore

from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    device: str = "cpu"  # cpu|cuda|mps
    outdir: str = "results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Keep seeding consistent across scripts."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Apple MPS
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            # Some torch versions may not expose torch.mps.manual_seed
            pass

def _jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    """Convert argparse Namespace to a JSON-serializable dict.

    Argparse stores the selected subcommand handler in `args.func`, which is a
    function and not JSON-serializable. We drop it. Also convert Path to str.
    """
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if k == "func":
            continue
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# Part A: VAE on binarized MNIST with different priors
# -----------------------------------------------------------------------------

def _project_to_2d(z: torch.Tensor) -> torch.Tensor:
    """Project latent samples to 2D (identity if dim=2, otherwise PCA).

    We depend on scikit-learn only when latent_dim > 2.
    """
    if z.shape[1] == 2:
        return z

    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for PCA when latent_dim > 2. Install it with: pip install scikit-learn"
        ) from e

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z.detach().cpu().numpy())
    return torch.from_numpy(z_2d).float()


@torch.no_grad()
def _sample_aggregate_posterior(
    model: vae_bernoulli.VAE,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_points: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ONE z ~ q(z|x) per datapoint (aggregate posterior approximation)."""
    model.eval()
    zs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
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


@torch.no_grad()
def plot_prior_vs_aggregate(
    model: vae_bernoulli.VAE,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: str,
    max_points: int = 10_000,
    n_prior: int = 10_000,
) -> None:
    """Plot prior samples and aggregate posterior samples in the same 2D projection."""
    import matplotlib.pyplot as plt

    # Aggregate posterior samples
    z_post, y = _sample_aggregate_posterior(model, data_loader, device, max_points=max_points)

    # Prior samples
    z_prior = model.prior().sample((n_prior,)).cpu()

    # Common 2D projection (fit PCA on concatenated samples)
    z_all = torch.cat([z_prior, z_post], dim=0)
    z2_all = _project_to_2d(z_all)

    z2_prior = z2_all[: z_prior.shape[0]]
    z2_post = z2_all[z_prior.shape[0] :]

    plt.figure(figsize=(8, 6))
    plt.scatter(z2_prior[:, 0], z2_prior[:, 1], s=6, alpha=0.25, label="Prior p(z)")
    sc = plt.scatter(
        z2_post[:, 0], z2_post[:, 1], c=y, s=6, alpha=0.75, label="Aggregate posterior q(z)"
    )
    plt.colorbar(sc, label="MNIST class (posterior samples)")
    plt.title("Prior vs Aggregate Posterior (2D projection)")
    plt.xlabel("z[0] / PC1")
    plt.ylabel("z[1] / PC2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

class _FlowDistributionAdapter:
    """Adapter to provide `.log_prob(z)` and `.sample(sample_shape)`."""

    def __init__(self, flow_model: "flow.Flow"):
        self.flow_model = flow_model

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow_model.log_prob(z)  # (B,)

    def sample(self, sample_shape: Tuple[int, ...]) -> torch.Tensor:
        if len(sample_shape) != 1:
            raise ValueError("Flow prior adapter expects sample_shape like (N,) only")
        return self.flow_model.sample(sample_shape[0])


class FlowPrior(nn.Module):
    """Flow-based prior p(z) built from the Week 2 flow implementation."""

    def __init__(self, flow_model: "flow.Flow"):
        super().__init__()
        self.flow_model = flow_model

    def forward(self):
        return _FlowDistributionAdapter(self.flow_model)


def build_flow_for_latent_dim(
    D: int,
    device: torch.device,
    num_layers: int = 8,
    hidden: int = 256,
) -> "flow.Flow":
    """Create a simple RealNVP-style flow for a given dimensionality."""
    transformations = []

    # Alternating masks
    m = torch.zeros(D)
    m[: D // 2] = 1.0

    for _ in range(num_layers):
        scale_net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D),
            nn.Tanh(),  # stability for scales
        )
        translation_net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D),
        )
        transformations.append(flow.MaskedCouplingLayer(scale_net, translation_net, m.clone()))
        m = 1.0 - m

    base = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros(D, device=device),
            torch.ones(D, device=device),
        ),
        1,
    )
    return flow.Flow(transformations, base)


def build_vae_bernoulli(
    latent_dim: int,
    prior_kind: str,
    mog_components: int,
    flow_ckpt: Optional[Path],
    device: torch.device,
) -> vae_bernoulli.VAE:
    """Build the Bernoulli VAE for Part A."""
    encoder_net, decoder_net = vae_bernoulli.build_default_mlp(latent_dim)
    encoder = vae_bernoulli.GaussianEncoder(encoder_net)
    decoder = vae_bernoulli.BernoulliDecoder(decoder_net)

    if prior_kind == "gaussian":
        prior = vae_bernoulli.GaussianPrior(latent_dim)
    elif prior_kind == "mog":
        prior = vae_bernoulli.MoGPrior(latent_dim, n_components=mog_components, trainable=True)
    elif prior_kind == "flow":
        # Trainable flow-based prior (optimized jointly with the VAE).
        # We initialize a flow and DO NOT require a checkpoint.
        # If --flow-ckpt is provided, we initialize from it but keep it trainable.
        flow_model = build_flow_for_latent_dim(latent_dim, device=device)
        if flow_ckpt is not None:
            flow_model.load_state_dict(torch.load(flow_ckpt, map_location=device))
        flow_model.to(device)
        prior = FlowPrior(flow_model)
    else:
        raise ValueError(f"Unknown prior_kind: {prior_kind}")

    model = vae_bernoulli.VAE(prior=prior, decoder=decoder, encoder=encoder).to(device)
    return model


def run_part_a_train(args: argparse.Namespace) -> None:
    cfg = RunConfig(seed=args.seed, device=args.device, outdir=args.outdir)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    outdir = Path(cfg.outdir) / "partA" / f"prior={args.prior}" / f"seed={cfg.seed}"
    ensure_dir(outdir)

    # Data
    train_loader, test_loader = vae_bernoulli.build_mnist_loaders(args.batch_size, binarized=True)

    # Model
    flow_ckpt = Path(args.flow_ckpt) if args.flow_ckpt else None
    model = build_vae_bernoulli(
        latent_dim=args.latent_dim,
        prior_kind=args.prior,
        mog_components=args.mog_components,
        flow_ckpt=flow_ckpt,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    vae_bernoulli.train(
        model, optimizer, train_loader, epochs=args.epochs, device=device, n_samples=args.elbo_samples
    )

    # Save weights
    torch.save(model.state_dict(), outdir / "model.pt")

    # Evaluate test ELBO
    test_elbo = vae_bernoulli.evaluate_average_elbo(
        model, test_loader, device, n_samples=args.elbo_samples
    )

    metrics = {
        "test_elbo": test_elbo,
        "seed": cfg.seed,
        "latent_dim": args.latent_dim,
        "prior": args.prior,
        "mog_components": args.mog_components,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "elbo_samples": args.elbo_samples,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Posterior plot
    vae_bernoulli.plot_aggregate_posterior(
        model=model,
        data_loader=test_loader,
        device=device,
        out_path=str(outdir / "aggregate_posterior.png"),
        max_points=args.max_posterior_points,
    )
    
    # Required by the project: plot of the prior and the aggregate posterior
    plot_prior_vs_aggregate(
        model=model,
        data_loader=test_loader,
        device=device,
        out_path=str(outdir / "prior_vs_aggregate.png"),
        max_points=args.max_posterior_points,
        n_prior=args.max_posterior_points,
    )

    # Samples
    from torchvision.utils import save_image

    with torch.no_grad():
        samples = model.sample(n_samples=64).cpu()
        save_image(samples.view(64, 1, 28, 28), str(outdir / "samples.png"))

    print(f"[PartA][train] Saved: {outdir}")
    print(f"[PartA][train] Test ELBO: {test_elbo:.4f}")


# -----------------------------------------------------------------------------
# Part B: DDPM + FID
# -----------------------------------------------------------------------------

# --- Part A sweep helpers (top-level) ---
def _mean_std(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    t = torch.tensor(values, dtype=torch.float32)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    return mean, std


def run_part_a_sweep(args: argparse.Namespace) -> None:
    """Run Part A for multiple seeds and write summary.json with mean/std."""
    out_root = Path(args.outdir) / "partA" / f"prior={args.prior}"
    ensure_dir(out_root)

    elbos: List[float] = []

    for s in args.seeds:
        sub_args = argparse.Namespace(**vars(args))
        sub_args.seed = int(s)
        run_part_a_train(sub_args)

        metrics_path = out_root / f"seed={int(s)}" / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        elbos.append(float(metrics["test_elbo"]))

    mean, std = _mean_std(elbos)

    summary = {
        "prior": args.prior,
        "latent_dim": args.latent_dim,
        "mog_components": args.mog_components,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "elbo_samples": args.elbo_samples,
        "seeds": [int(s) for s in args.seeds],
        "test_elbo_values": elbos,
        "test_elbo_mean": mean,
        "test_elbo_std": std,
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[PartA][sweep] Summary written to:", out_root / "summary.json")
    print(f"[PartA][sweep] Test ELBO mean ± std: {mean:.4f} ± {std:.4f}")


###############################################################################
# Part B: DDPM + FID
###############################################################################

# -----------------------------------------------------------------------------
# Part B.1: beta-VAE on standard MNIST (Gaussian likelihood)
# -----------------------------------------------------------------------------

class BetaVAE(nn.Module):
    """A VAE with Gaussian likelihood and a beta-weighted KL term.

    Uses:
      - Gaussian encoder q(z|x)
      - Gaussian prior p(z) = N(0,I)
      - Gaussian decoder p(x|z)

    Training objective (to minimize):
      L = -E_q[log p(x|z)] + beta * KL(q(z|x) || p(z))

    We keep it simple and re-use the Week 1 MLPs.
    """

    def __init__(
        self,
        latent_dim: int,
        fixed_sigma: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        encoder_net, decoder_net = vae_bernoulli.build_default_mlp(latent_dim)
        self.encoder = vae_bernoulli.GaussianEncoder(encoder_net)
        self.prior = vae_bernoulli.GaussianPrior(latent_dim)

        # Decoder outputs mean in [-inf, inf]; data is scaled to [-1,1].
        # We use a fixed sigma for stability/reproducibility.
        self.decoder = vae_bernoulli.GaussianDecoder(
            decoder_net=decoder_net,
            fixed_sigma=fixed_sigma,
            learn_per_pixel_sigma=False,
            learn_global_sigma=False,
        )

    def loss(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Return per-sample loss (B,) for a batch."""
        q = self.encoder(x)
        pz = self.prior()

        z = q.rsample((n_samples,))  # (S,B,M)
        z_flat = z.view(-1, z.shape[-1])
        x_rep = x.repeat((n_samples,) + (1,) * (x.ndim - 1))
        # NOTE:
        #   The Week 1 GaussianDecoder returns a Normal distribution whose batch/event
        #   shape matches image-like tensors (B, 28, 28). Our training data for the
        #   MLP encoder is flattened to (B, 784), so we reshape ONLY for the
        #   likelihood evaluation.
        x_rep_img = x_rep.view(-1, 28, 28)

        log_px = self.decoder(z_flat).log_prob(x_rep_img).view(n_samples, -1)  # (S,B)
        log_pz = pz.log_prob(z_flat).view(n_samples, -1)  # (S,B)
        log_qz = q.log_prob(z_flat).view(n_samples, -1)  # (S,B)

        # -E_q log p(x|z)
        recon = (-log_px).mean(dim=0)  # (B,)

        # KL(q||p) = E_q[log q - log p]
        kl = (log_qz - log_pz).mean(dim=0)  # (B,)

        return recon + self.beta * kl

    @torch.no_grad()
    def sample(self, n: int, return_mean: bool = True) -> torch.Tensor:
        """Sample x from the model.

        Returns:
          (n, 784) tensor in the SAME scale as training data ([-1,1]).
        """
        self.eval()
        device = next(self.parameters()).device
        z = self.prior().sample((n,)).to(device)
        px = self.decoder(z)
        x = px.mean if return_mean else px.sample()
        return x


def _mnist_continuous_loaders(batch_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """MNIST loaders for Part B.

    - Loads x in [0,1]
    - Dequantizes: x + U(0, 1/255)
    - Scales to [-1,1]
    - Flattens to (784,)
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand_like(x) / 255.0),
            transforms.Lambda(lambda x: (x.squeeze() - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train = datasets.MNIST("data/", train=True, download=True, transform=transform)
    test = datasets.MNIST("data/", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


@torch.no_grad()
def _collect_latents(
    model: BetaVAE,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_points: int = 100_000,
    use_mean: bool = True,
) -> torch.Tensor:
    """Collect latent vectors z from q(z|x) for a dataset."""
    model.eval()
    zs: List[torch.Tensor] = []
    count = 0

    for batch in data_loader:
        x = batch[0].to(device)
        q = model.encoder(x)
        z = q.mean if use_mean else q.rsample()
        zs.append(z.detach().cpu())
        count += x.shape[0]
        if count >= max_points:
            break

    return torch.cat(zs, dim=0)[:max_points]


def run_part_b_beta_vae_train(args: argparse.Namespace) -> None:
    """Train beta-VAE on standard MNIST (Gaussian likelihood) and optionally compute FID + samples."""
    cfg = RunConfig(seed=args.seed, device=args.device, outdir=args.outdir)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    outdir = Path(cfg.outdir) / "partB" / "beta_vae" / f"beta={args.beta:g}" / f"seed={cfg.seed}"
    ensure_dir(outdir)

    train_loader, test_loader = _mnist_continuous_loaders(batch_size=args.batch_size)

    model = BetaVAE(latent_dim=args.latent_dim, fixed_sigma=args.fixed_sigma, beta=args.beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for batch in tqdm(train_loader, desc=f"[beta-VAE] epoch {epoch+1}/{args.epochs}"):
            x = batch[0].to(device)
            loss = model.loss(x, n_samples=args.elbo_samples).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), outdir / "model.pt")

    # Save a small sample grid
    from torchvision.utils import save_image

    with torch.no_grad():
        x_s = model.sample(64, return_mean=True).cpu()
        x_s = (x_s.view(-1, 1, 28, 28) / 2.0 + 0.5).clamp(0.0, 1.0)
        save_image(x_s, str(outdir / "samples.png"))

    # Quick metrics: average test loss
    model.eval()
    total = 0.0
    n = 0
    for batch in test_loader:
        x = batch[0].to(device)
        l = model.loss(x, n_samples=args.elbo_samples).mean().item()
        total += l * x.shape[0]
        n += x.shape[0]
    avg_test_loss = total / max(n, 1)

    metrics = {
        "seed": cfg.seed,
        "beta": float(args.beta),
        "latent_dim": args.latent_dim,
        "fixed_sigma": float(args.fixed_sigma),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "elbo_samples": args.elbo_samples,
        "avg_test_loss": float(avg_test_loss),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[PartB][beta-VAE] Saved: {outdir}")
    print(f"[PartB][beta-VAE] Avg test loss: {avg_test_loss:.4f}")


# -----------------------------------------------------------------------------
# Part B.2: Latent DDPM (train DDPM in VAE latent space)
# -----------------------------------------------------------------------------

@torch.no_grad()
def _decode_latents_to_images(model: BetaVAE, z: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Decode latent vectors to MNIST images in [0,1], shape (N,1,28,28)."""
    model.eval()
    z = z.to(device)
    px = model.decoder(z)
    x = px.mean  # use mean for cleaner samples
    x = x.detach().cpu().view(-1, 1, 28, 28)
    x = (x / 2.0 + 0.5).clamp(0.0, 1.0)
    return x


def run_part_b_latent_ddpm_train(args: argparse.Namespace) -> None:
    """Train a DDPM in the latent space of a (pretrained) beta-VAE and compute samples/FID/timing/latent plots."""
    cfg = RunConfig(seed=args.seed, device=args.device, outdir=args.outdir)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    outdir = Path(cfg.outdir) / "partB" / "latent_ddpm" / f"beta={args.beta:g}" / f"seed={cfg.seed}"
    ensure_dir(outdir)
    (outdir / "args.json").write_text(json.dumps(_jsonable_args(args), indent=2))

    # Load beta-VAE
    vae_dir = Path(cfg.outdir) / "partB" / "beta_vae" / f"beta={args.beta:g}" / f"seed={cfg.seed}"
    ckpt = vae_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Could not find beta-VAE checkpoint at {ckpt}. Train it first with partB_beta_vae_train."
        )

    train_loader, test_loader = _mnist_continuous_loaders(batch_size=args.batch_size)

    beta_vae = BetaVAE(latent_dim=args.latent_dim, fixed_sigma=args.fixed_sigma, beta=args.beta).to(device)
    beta_vae.load_state_dict(torch.load(ckpt, map_location=device))
    beta_vae.eval()

    # Build latent dataset
    z_train = _collect_latents(beta_vae, train_loader, device, max_points=args.latent_points, use_mean=True)
    z_test = _collect_latents(beta_vae, test_loader, device, max_points=min(args.latent_points, 50_000), use_mean=True)

    z_train_loader = torch.utils.data.DataLoader(z_train, batch_size=args.latent_batch_size, shuffle=True)

    # Train DDPM on z
    D = args.latent_dim
    net = ddpm.FcNetwork(D, args.hidden)
    z_ddpm = ddpm.DDPM(network=net, T=args.T).to(device)
    opt = torch.optim.Adam(z_ddpm.parameters(), lr=args.lr)

    z_ddpm.train()
    for epoch in range(args.epochs):
        for z in tqdm(z_train_loader, desc=f"[latent-DDPM] epoch {epoch+1}/{args.epochs}"):
            z = z.to(device)
            loss = z_ddpm.negative_elbo(z).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    torch.save(z_ddpm.state_dict(), outdir / "latent_ddpm.pt")

    # Sample latent DDPM -> decode
    from torchvision.utils import save_image

    # Sample latent DDPM -> decode (in chunks to avoid MPS OOM)
    t0 = time.perf_counter()

    z_chunks = []
    remaining = args.fid_gen
    chunk = min(512, args.fid_gen)  # adjust if needed (256 if still OOM)

    z_ddpm.eval()
    with torch.no_grad():
        while remaining > 0:
            n = min(chunk, remaining)
            z_part = z_ddpm.sample((n, D)).detach().cpu()
            z_chunks.append(z_part)
            remaining -= n

    z_s = torch.cat(z_chunks, dim=0)  # (fid_gen, D)

    imgs = _decode_latents_to_images(beta_vae, z_s, device=device)

    t1 = time.perf_counter()
    seconds = t1 - t0

    # Save a small grid
    save_image(imgs[:64], str(outdir / "samples.png"))

    # Real images for FID (from test set)
    real_images = []
    max_real = args.fid_real
    for batch in test_loader:
        real_images.append(batch[0])
        if sum(b.shape[0] for b in real_images) >= max_real:
            break
    real = torch.cat(real_images, dim=0)[:max_real]
    real = (real.view(-1, 1, 28, 28) / 2.0 + 0.5).clamp(0.0, 1.0)

    fid_score = compute_fid_wrapper(
        real_images=real,
        gen_images=imgs[: args.fid_gen],
        classifier_ckpt=Path(args.mnist_classifier),
        device=device,
    )

    # Latent space plot: prior vs agg posterior vs latent DDPM
    try:
        import matplotlib.pyplot as plt

        z_prior = beta_vae.prior().sample((min(10_000, z_test.shape[0]),)).cpu()
        z_post = z_test[: min(10_000, z_test.shape[0])].cpu()
        z_flow = z_s[: min(10_000, z_s.shape[0])].cpu()

        z_all = torch.cat([z_prior, z_post, z_flow], dim=0)
        z2 = _project_to_2d(z_all)

        n1 = z_prior.shape[0]
        n2 = z_post.shape[0]
        z2_prior = z2[:n1]
        z2_post = z2[n1 : n1 + n2]
        z2_latddpm = z2[n1 + n2 :]

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.scatter(z2_prior[:, 0], z2_prior[:, 1], s=4, alpha=0.5)
        plt.title("Prior p(z)")
        plt.subplot(1, 3, 2)
        plt.scatter(z2_post[:, 0], z2_post[:, 1], s=4, alpha=0.5)
        plt.title("Aggregate posterior q(z)")
        plt.subplot(1, 3, 3)
        plt.scatter(z2_latddpm[:, 0], z2_latddpm[:, 1], s=4, alpha=0.5)
        plt.title("Latent DDPM p_θ(z)")
        plt.tight_layout()
        plt.savefig(str(outdir / "latent_space.png"), dpi=200)
        plt.close()
    except Exception as e:
        print("[PartB][latent-DDPM] Warning: could not save latent_space.png:", e)

    metrics = {
        "seed": cfg.seed,
        "beta": float(args.beta),
        "latent_dim": args.latent_dim,
        "fixed_sigma": float(args.fixed_sigma),
        "T": int(args.T),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "latent_points": int(args.latent_points),
        "latent_batch_size": int(args.latent_batch_size),
        "hidden": int(args.hidden),
        "lr": float(args.lr),
        "fid_real": int(args.fid_real),
        "fid_gen": int(args.fid_gen),
        "fid": float(fid_score),
        "sample_time_seconds_for_fid_gen": float(seconds),
        "samples_per_second": float(args.fid_gen / max(seconds, 1e-9)),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[PartB][latent-DDPM] Saved: {outdir}")
    print(f"[PartB][latent-DDPM] FID: {fid_score:.4f}")
    print(f"[PartB][latent-DDPM] Sampling time ({args.fid_gen}): {seconds:.3f}s")

@torch.no_grad()
def sample_mnist_from_ddpm(model: ddpm.DDPM, n: int) -> torch.Tensor:
    """Sample MNIST from a DDPM model. Returns (n,1,28,28) in [0,1]."""
    model.eval()
    x = model.sample((n, 28 * 28)).to("cpu")
    x = x.view(-1, 1, 28, 28)
    x = (x / 2.0 + 0.5).clamp(0.0, 1.0)  # ddpm trains on [-1,1]
    return x


def compute_fid_wrapper(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    classifier_ckpt: Path,
    device: torch.device,
) -> float:
    """Compute FID using provided fid.py (signature-robust; no need to edit fid.py)."""
    # IMPORTANT (Apple MPS / NNPack): fid.py uses a small convnet classifier and may hit
    # NNPack CPU code paths that require float32 + contiguous inputs.
    # To avoid "Mismatched Tensor types in NNPack convolutionOutput", force CPU float32.
    real_images = real_images.detach().to("cpu").contiguous().float()
    gen_images = gen_images.detach().to("cpu").contiguous().float()
    device = torch.device("cpu")

    fn = fid.compute_fid
    sig = inspect.signature(fn)
    kwargs: Dict[str, object] = {}

    for name in sig.parameters:
        if name in {"real", "real_images", "x_real"}:
            kwargs[name] = real_images
        elif name in {"gen", "gen_images", "x_gen", "fake", "fake_images"}:
            kwargs[name] = gen_images
        elif name in {"classifier_path", "classifier_ckpt", "model_path", "ckpt_path"}:
            kwargs[name] = str(classifier_ckpt)
        elif name in {"device"}:
            kwargs[name] = "cpu"
        elif name in {"batch_size", "bs"}:
            kwargs[name] = 256

    try:
        out = fn(**kwargs)
    except TypeError:
        out = fn(real_images, gen_images)

    if isinstance(out, (float, int)):
        return float(out)
    if torch.is_tensor(out):
        return float(out.item())
    raise TypeError(f"Unexpected compute_fid return type: {type(out)}")


def run_part_b_ddpm_train(args: argparse.Namespace) -> None:
    cfg = RunConfig(seed=args.seed, device=args.device, outdir=args.outdir)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    outdir = Path(cfg.outdir) / "partB" / "ddpm" / f"seed={cfg.seed}"
    ensure_dir(outdir)

    # Data: dequantize and scale to [-1,1]
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255.0),
            transforms.Lambda(lambda x: (x.squeeze() - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_data = datasets.MNIST("data/", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data/", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    D = 28 * 28

    if args.net == "unet":
        net: nn.Module = unet.Unet()
    else:
        net = ddpm.FcNetwork(D, args.hidden)

    model = ddpm.DDPM(network=net, T=args.T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    model.train()
    for epoch in range(args.epochs):
        for batch in tqdm(train_loader, desc=f"[DDPM] epoch {epoch+1}/{args.epochs}"):
            x = batch[0].to(device)
            loss = model.negative_elbo(x).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # Save
    torch.save(model.state_dict(), outdir / "model.pt")

    # Samples + time
    t0 = time.perf_counter()
    samples = sample_mnist_from_ddpm(model, n=64)
    t1 = time.perf_counter()
    seconds = t1 - t0

    from torchvision.utils import save_image

    save_image(samples, str(outdir / "samples.png"))

    # Real batch for FID
    real_images = []
    max_real = args.fid_real
    for batch in test_loader:
        real_images.append(batch[0])
        if sum(b.shape[0] for b in real_images) >= max_real:
            break

    real = torch.cat(real_images, dim=0)[:max_real]
    real = (real.view(-1, 1, 28, 28) / 2.0 + 0.5).clamp(0.0, 1.0)

    gen = sample_mnist_from_ddpm(model, n=args.fid_gen)

    fid_score = compute_fid_wrapper(
        real_images=real,
        gen_images=gen,
        classifier_ckpt=Path(args.mnist_classifier),
        device=device,
    )

    metrics = {
        "seed": cfg.seed,
        "T": args.T,
        "net": args.net,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sample_time_seconds_for_64": seconds,
        "fid_real": int(args.fid_real),
        "fid_gen": int(args.fid_gen),
        "fid": fid_score,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[PartB][DDPM] Saved: {outdir}")
    print(f"[PartB][DDPM] FID: {fid_score:.4f}")
    print(f"[PartB][DDPM] Sampling time (64): {seconds:.3f}s")



# -----------------------------------------------------------------------------
# Part C: Export required code snippets (1-page report appendix)
# -----------------------------------------------------------------------------

def _safe_getsource(obj) -> str:
    """Get source code for an object, with a helpful fallback."""
    try:
        return inspect.getsource(obj)
    except OSError:
        return f"# [Could not retrieve source for {obj}]\n"


def _write_snippet(md_lines: List[str], title: str, code: str) -> None:
    md_lines.append(f"## {title}\n")
    md_lines.append("```python\n")
    md_lines.append(code.rstrip() + "\n")
    md_lines.append("```\n\n")


def run_part_c_export_snippets(args: argparse.Namespace) -> None:
    """Export the exact code snippets requested in Part C to a Markdown file.

    This does NOT affect grading directly, but it makes it easy to copy/paste into
    the 1-page 'code snippets' part of the report template.

    The project statement asks for snippets covering:
      - VAE ELBO with a non-Gaussian prior
      - Central parts of the Flow implementation
      - DDPM negative ELBO
      - DDPM sampling
      - Central parts of latent DDPM training and sampling
    """
    outdir = Path(args.outdir) / "partC"
    ensure_dir(outdir)

    md: List[str] = []
    md.append("# 02460 Mini-project 1 — Part C: Code snippets\n\n")
    md.append("Copy/paste the relevant blocks into the report template.\n\n")

    # --- VAE ELBO (non-Gaussian prior) ---
    # We reference the generic ELBO implementation in the Week 1 VAE.
    md.append("## VAE ELBO (works with non-Gaussian priors)\n\n")
    md.append("```python\n")
    md.append(_safe_getsource(vae_bernoulli.VAE.elbo).rstrip() + "\n")
    md.append("```\n\n")

    # --- Flow core: coupling layer forward/inverse ---
    _write_snippet(md, "Flow core: MaskedCouplingLayer.forward", _safe_getsource(flow.MaskedCouplingLayer.forward))
    _write_snippet(md, "Flow core: MaskedCouplingLayer.inverse", _safe_getsource(flow.MaskedCouplingLayer.inverse))

    # --- DDPM negative ELBO + sampling ---
    _write_snippet(md, "DDPM training objective: DDPM.negative_elbo", _safe_getsource(ddpm.DDPM.negative_elbo))
    _write_snippet(md, "DDPM sampling: DDPM.sample", _safe_getsource(ddpm.DDPM.sample))

    # --- Latent DDPM: central training + sampling pieces (this file) ---
    _write_snippet(md, "Latent DDPM: collect latents", _safe_getsource(_collect_latents))
    _write_snippet(md, "Latent DDPM: decode latents to images", _safe_getsource(_decode_latents_to_images))
    _write_snippet(md, "Latent DDPM: training + sampling runner", _safe_getsource(run_part_b_latent_ddpm_train))

    out_path = outdir / "snippets.md"
    out_path.write_text("".join(md))

    print(f"[PartC] Wrote snippets to: {out_path}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # Part A
    pa = sub.add_parser("partA_train", help="Train Bernoulli VAE on binarized MNIST with chosen prior")
    pa.add_argument("--prior", choices=["gaussian", "mog", "flow"], default="gaussian")
    pa.add_argument("--latent-dim", type=int, default=10)
    pa.add_argument("--mog-components", type=int, default=10)
    pa.add_argument("--flow-ckpt", type=str, default=None, help="Optional init checkpoint for the flow prior")
    pa.add_argument("--epochs", type=int, default=10)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--elbo-samples", type=int, default=1)
    pa.add_argument("--max-posterior-points", type=int, default=10_000)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    pa.add_argument("--outdir", type=str, default="results")
    pa.set_defaults(func=run_part_a_train)
    
    # Part A sweep (multiple seeds -> mean/std)
    pas = sub.add_parser("partA_sweep", help="Run Part A training for multiple seeds and summarize mean/std")
    pas.add_argument("--prior", choices=["gaussian", "mog", "flow"], default="gaussian")
    pas.add_argument("--latent-dim", type=int, default=10)
    pas.add_argument("--mog-components", type=int, default=10)
    pas.add_argument("--flow-ckpt", type=str, default=None, help="Optional init checkpoint for the flow prior")
    pas.add_argument("--epochs", type=int, default=10)
    pas.add_argument("--batch-size", type=int, default=128)
    pas.add_argument("--lr", type=float, default=1e-3)
    pas.add_argument("--elbo-samples", type=int, default=1)
    pas.add_argument("--max-posterior-points", type=int, default=10_000)
    pas.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="List of seeds to run")
    pas.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    pas.add_argument("--outdir", type=str, default="results")
    pas.set_defaults(func=run_part_a_sweep)

    # Part B: DDPM
    pb = sub.add_parser("partB_ddpm_train", help="Train DDPM on MNIST and compute samples/FID")
    pb.add_argument("--net", choices=["unet", "fc"], default="unet")
    pb.add_argument("--hidden", type=int, default=128, help="Only used for fc network")
    pb.add_argument("--T", type=int, default=1000)
    pb.add_argument("--epochs", type=int, default=10)
    pb.add_argument("--batch-size", type=int, default=128)
    pb.add_argument("--lr", type=float, default=1e-3)
    pb.add_argument("--fid-real", type=int, default=10_000)
    pb.add_argument("--fid-gen", type=int, default=10_000)
    pb.add_argument("--mnist-classifier", type=str, default=str(THIS_DIR / "mnist_classifier.pth"))
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    pb.add_argument("--outdir", type=str, default="results")
    pb.set_defaults(func=run_part_b_ddpm_train)

    # Part B: beta-VAE (Gaussian likelihood)
    bv = sub.add_parser("partB_beta_vae_train", help="Train beta-VAE on standard MNIST (Gaussian likelihood)")
    bv.add_argument("--beta", type=float, default=1.0)
    bv.add_argument("--latent-dim", type=int, default=10)
    bv.add_argument("--fixed-sigma", type=float, default=0.1)
    bv.add_argument("--epochs", type=int, default=10)
    bv.add_argument("--batch-size", type=int, default=128)
    bv.add_argument("--lr", type=float, default=1e-3)
    bv.add_argument("--elbo-samples", type=int, default=1)
    bv.add_argument("--seed", type=int, default=0)
    bv.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    bv.add_argument("--outdir", type=str, default="results")
    bv.set_defaults(func=run_part_b_beta_vae_train)

    # Part B: latent DDPM
    ld = sub.add_parser("partB_latent_ddpm_train", help="Train latent DDPM using a pretrained beta-VAE and compute FID")
    ld.add_argument("--beta", type=float, default=1.0)
    ld.add_argument("--latent-dim", type=int, default=10)
    ld.add_argument("--fixed-sigma", type=float, default=0.1)
    ld.add_argument("--T", type=int, default=1000)
    ld.add_argument("--epochs", type=int, default=10)
    ld.add_argument("--batch-size", type=int, default=128)
    ld.add_argument("--latent-points", type=int, default=100_000)
    ld.add_argument("--latent-batch-size", type=int, default=512)
    ld.add_argument("--hidden", type=int, default=256)
    ld.add_argument("--lr", type=float, default=1e-3)
    ld.add_argument("--fid-real", type=int, default=10_000)
    ld.add_argument("--fid-gen", type=int, default=10_000)
    ld.add_argument("--mnist-classifier", type=str, default=str(THIS_DIR / "mnist_classifier.pth"))
    ld.add_argument("--seed", type=int, default=0)
    ld.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ld.add_argument("--outdir", type=str, default="results")
    ld.set_defaults(func=run_part_b_latent_ddpm_train)

    # Part C: export code snippets
    pc = sub.add_parser("partC_export_snippets", help="Export the required Part C code snippets to results/partC/snippets.md")
    pc.add_argument("--outdir", type=str, default="results")
    pc.set_defaults(func=run_part_c_export_snippets)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()