import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torchinfo import summary
import sys
from argparse import ArgumentParser
import time

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

## Copy from https://github.com/yunjinli/TRASE/blob/master/gui_standalone.py
## ======= Deformable 3D Gaussian =======
def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class DeformNetwork(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=59,
        multires=10,
        is_blender=False,
        is_6dof=False,
        verbose=True,
    ):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch
        if verbose:
            print(f"self.t_multires: {self.t_multires}")
            print(f"multires: {multires}")
            print(f"time_input_ch: {time_input_ch}")
            print(f"xyz_input_ch: {xyz_input_ch}")
        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)
        
        return d_xyz, rotation, scaling
    

def export_deform_weights(weights_path, output_path):
    D=8
    W=256
    model = DeformNetwork(D=D, W=W, verbose=True)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    summary(model)
    
    weights_list = []
    
    for name, param in model.named_parameters():
        # Export as Float32
        p = param.detach().numpy().astype(np.float32)
        weights_list.append(p.flatten())
        
    all_weights = np.concatenate(weights_list)
    
    all_weights.tofile(output_path)
    
    print(f"Exported {len(all_weights)} weights")

def run_deformnetwork_benchmark(
    *,
    iters: int = 50,
    batch_size: int = 16384,
    device: str = "cpu",
    weights_path: str | None = None,
):
    """
    Runs DeformNetwork forward pass in a loop to sanity-check performance.

    Inputs:
      - x: (B, 3)
      - t: (B, 1)
    """
    dev = torch.device(device)
    model = DeformNetwork(D=8, W=256, verbose=False).to(dev)
    model.eval()

    def _sync():
        # Make per-iteration timing accurate on async backends.
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elif dev.type == "mps":
            torch.mps.synchronize()

    if weights_path is not None:
        state = torch.load(weights_path, map_location=dev)
        model.load_state_dict(state)

    x = torch.rand(batch_size, 3, device=dev, dtype=torch.float32)
    t = torch.rand(batch_size, 1, device=dev, dtype=torch.float32)

    # Small warmup so the tqdm loop is representative.
    with torch.inference_mode():
        for _ in range(min(5, iters)):
            _ = model(x, t)

        iterator = range(iters)
        if tqdm is not None:
            iterator = tqdm(iterator, total=iters, desc=f"DeformNetwork B={batch_size} ({device})")

        per_iter_s: list[float] = []
        _sync()
        start = time.perf_counter()
        for i in iterator:
            _sync()
            t0 = time.perf_counter()
            _ = model(x, t)
            _sync()
            t1 = time.perf_counter()

            dt = t1 - t0
            per_iter_s.append(dt)
            if tqdm is not None:
                # show instantaneous + running average
                avg_ms = 1000.0 * (sum(per_iter_s) / len(per_iter_s))
                iterator.set_postfix_str(f"{dt*1000.0:.2f} ms/batch (avg {avg_ms:.2f})")

        _sync()
        end = time.perf_counter()

    total_s = end - start
    if total_s > 0:
        avg_ms = 1000.0 * (sum(per_iter_s) / len(per_iter_s)) if per_iter_s else float("nan")
        print(
            f"Done: {iters} iters, batch={batch_size}, device={device}, "
            f"{total_s:.3f}s total ({iters/total_s:.2f} it/s), avg {avg_ms:.2f} ms/batch"
        )
    else:
        print(f"Done: {iters} iters, batch={batch_size}, device={device}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Export deform weights, or run a DeformNetwork benchmark loop.")
    parser.add_argument("--model", default="deform.pth", type=str, help="Path to deform.pth. (Default: deform.pth)")
    parser.add_argument("--output", default="weights.bin", type=str, help="Output weights path. (Default: weights.bin)")
    parser.add_argument("--benchmark", action="store_true", help="Run a forward-pass benchmark loop instead of exporting weights.")
    parser.add_argument("--iters", default=50, type=int, help="Benchmark iterations. (Default: 50)")
    parser.add_argument("--batch-size", default=16384, type=int, help="Benchmark batch size. (Default: 16384)")
    parser.add_argument("--device", default="cpu", type=str, help="Benchmark device, e.g. cpu / mps / cuda. (Default: cpu)")
    args = parser.parse_args()
    if args.benchmark:
        run_deformnetwork_benchmark(
            iters=args.iters,
            batch_size=args.batch_size,
            device=args.device,
            weights_path=args.model if args.model else None,
        )
    else:
        export_deform_weights(weights_path=args.model, output_path=args.output)
