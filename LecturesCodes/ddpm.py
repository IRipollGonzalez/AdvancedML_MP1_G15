# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        # Diffusion schedule constants (buffers, not trainable parameters)
        beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1.0 - beta
        alpha_cumprod = alpha.cumprod(dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
    
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

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """Train a DDPM model by minimizing the negative ELBO (denoising objective)."""
    model.train()

    for epoch in range(epochs):
        epoch_bar = tqdm(data_loader, desc=f"Training epoch {epoch+1}/{epochs}")
        for x in epoch_bar:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard, mnist: MNIST} (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (toy defaults to 10000; for MNIST consider 128) (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--T', type=int, default=1000, metavar='N', help='number of diffusion steps (default: %(default)s)')
    parser.add_argument('--n-data', type=int, default=1000000, metavar='N', help='number of toy samples to generate for train/test (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate / load the data
    if args.data in ['tg', 'cb']:
        # Toy datasets: sample from the known distribution and scale to [-1, 1]
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x - 0.5) * 2.0

        n_data = args.n_data
        train_loader = torch.utils.data.DataLoader(
            transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True
        )
    else:
        # MNIST: load images in [0, 1], then scale to [-1, 1] and flatten to vectors of size 784.
        # This matches the provided U-Net forward() which expects flattened inputs.
        from torchvision import datasets, transforms

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x.squeeze() - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.view(-1)),  # (784,)
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', train=True, download=True, transform=mnist_transform),
            batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', train=False, download=True, transform=mnist_transform),
            batch_size=args.batch_size, shuffle=False
        )

    # Get the dimension of the dataset
    D = next(iter(train_loader)).shape[1]

    # Define the network
    if args.data in ['tg', 'cb']:
        num_hidden = 64
        network = FcNetwork(D, num_hidden)
    else:
        # For MNIST we use the provided U-Net architecture (expects flattened x and t of shape (B, 1))
        import unet
        network = unet.Unet()

    # Set the number of steps in the diffusion process
    T = args.T

    device = torch.device(args.device)

    # Define model
    model = DDPM(network, T=T).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            if args.data in ['tg', 'cb']:
                samples = (model.sample((10000, D))).cpu()
            else:
                # MNIST: sample 64 images as flattened vectors
                samples = (model.sample((64, D))).cpu()

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

        if args.data == 'mnist':
            # Save MNIST samples as an image grid
            from torchvision.utils import save_image
            samples_img = samples.view(-1, 1, 28, 28).clamp(0.0, 1.0)
            save_image(samples_img, args.samples)
            print(f"Saved MNIST samples to: {args.samples}")
            raise SystemExit(0)

        # Plot the density of the toy data and the model samples
        coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
        prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        ax.set_xlim(toy.xlim)
        ax.set_ylim(toy.ylim)
        ax.set_aspect('equal')
        fig.colorbar(im)
        plt.savefig(args.samples)
        plt.close()
