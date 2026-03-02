# 02460 Mini-project 1 — Part C: Code snippets

Copy/paste the relevant blocks into the report template.

## VAE ELBO (works with non-Gaussian priors)

```python
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
```

## Flow core: MaskedCouplingLayer.forward
```python
    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        # Split features using the binary mask (1 = frozen, 0 = transformed)
        # Shapes: (B, D)
        x = z
        mask = self.mask.view(1, -1)  # (1, D) for broadcasting

        x_masked = x * mask

        # Compute scale and translation for the transformed part
        s = self.scale_net(x_masked) * (1.0 - mask)
        t = self.translation_net(x_masked) * (1.0 - mask)

        # Stabilize: constrain scale via tanh (standard RealNVP trick)
        s = torch.tanh(s)

        # Forward transform (base -> data)
        y = x_masked + (1.0 - mask) * (x * torch.exp(s) + t)

        # Log-det Jacobian: sum over transformed dims
        log_det_J = (s).sum(dim=1)
        return y, log_det_J
```

## Flow core: MaskedCouplingLayer.inverse
```python
    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        mask = self.mask.view(1, -1)
        y = x

        y_masked = y * mask

        s = self.scale_net(y_masked) * (1.0 - mask)
        t = self.translation_net(y_masked) * (1.0 - mask)
        s = torch.tanh(s)

        # Inverse transform (data -> base)
        z = y_masked + (1.0 - mask) * ((y - t) * torch.exp(-s))

        # Inverse log-det is negative of forward
        log_det_J = (-s).sum(dim=1)
        return z, log_det_J
```

## DDPM training objective: DDPM.negative_elbo
```python
    def negative_elbo(self, x0: torch.Tensor) -> torch.Tensor:
        """Monte Carlo estimate of the DDPM negative ELBO (denoising objective).

        Implements:
            t ~ Uniform({1..T})
            eps ~ N(0, I)
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
            loss = ||eps - eps_theta(x_t, t)||^2

        Returns:
            Tensor of shape (B,) — per-sample loss.
        """
        if not torch.is_tensor(x0):
            raise TypeError(f"x0 must be a torch.Tensor, got {type(x0)}")

        device = x0.device
        B = x0.shape[0]

        # Sample random timesteps in {1,...,T}
        t = torch.randint(1, self.T + 1, (B,), device=device)

        # Gaussian noise
        eps = torch.randn_like(x0)

        # Get alpha_bar_t
        alpha_bar_t = self.alpha_cumprod[t - 1]  # shape (B,)

        # Broadcast to match x0 shape
        while alpha_bar_t.ndim < x0.ndim:
            alpha_bar_t = alpha_bar_t.view(-1, *([1] * (x0.ndim - 1)))

        # Forward diffusion
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps

        # Predict noise (our networks expect a normalized (B,1) float timestep feature)
        t_feat = (t.float() / float(self.T)).view(B, 1)
        eps_hat = self.network(x_t, t_feat)

        # MSE per sample
        loss = (eps_hat - eps) ** 2
        loss = loss.view(B, -1).mean(dim=1)

        return loss
```

## DDPM sampling: DDPM.sample
```python
    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T - 1, -1, -1):
            # Current timestep as integer in {1,...,T}
            t_int = t + 1
            t_batch = torch.full(
                (x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long
            )

            # Predict noise eps_theta(x_t, t)
            # FcNetwork expects t as (B,1) float feature
            t_feat = (t_batch.float() / float(self.T)).view(x_t.shape[0], 1)
            eps_hat = self.network(x_t, t_feat)

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]

            # Broadcast scalars to match x_t shape
            while beta_t.ndim < x_t.ndim:
                beta_t = beta_t.view(*([1] * x_t.ndim))
                alpha_t = alpha_t.view(*([1] * x_t.ndim))
                alpha_bar_t = alpha_bar_t.view(*([1] * x_t.ndim))

            # Reverse mean (DDPM update)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
            )

            if t > 0:
                z = torch.randn_like(x_t)

                # Posterior variance (DDPM): beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
                alpha_bar_prev = self.alpha_cumprod[t - 1]
                beta_tilde = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

                # Broadcast to match x_t shape
                while beta_tilde.ndim < x_t.ndim:
                    beta_tilde = beta_tilde.view(*([1] * x_t.ndim))

                x_t = mean + torch.sqrt(beta_tilde) * z
            else:
                x_t = mean

        return x_t
```

## Latent DDPM: collect latents
```python
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
```

## Latent DDPM: decode latents to images
```python
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
```

## Latent DDPM: training + sampling runner
```python
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
```

