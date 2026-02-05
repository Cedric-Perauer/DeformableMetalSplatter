#!/usr/bin/env python3
"""
Run the deformation network at a specific timestep and export the deformed point cloud as a PLY file.

Usage:
    python3 export_deformed_ply.py --ply point_cloud.ply --model deform.pth --time 0.5 --output deformed_0.5.ply
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R

# DeformNetwork implementation
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
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, 
                 is_blender=False, is_6dof=False, verbose=False):
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
        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )
        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)
        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)
        return d_xyz, rotation, scaling


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (wxyz format)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack([w, x, y, z], axis=-1)


def normalize_quaternion(q):
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + 1e-12)


def load_ply(ply_path):
    """Load a 3DGS PLY file and extract splat properties."""
    print(f"Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Extract positions
    x = np.array(vertex['x'], dtype=np.float32)
    y = np.array(vertex['y'], dtype=np.float32)
    z = np.array(vertex['z'], dtype=np.float32)
    positions = np.stack([x, y, z], axis=-1)
    
    # Extract scales (log space)
    scale_0 = np.array(vertex['scale_0'], dtype=np.float32)
    scale_1 = np.array(vertex['scale_1'], dtype=np.float32)
    scale_2 = np.array(vertex['scale_2'], dtype=np.float32)
    scales = np.stack([scale_0, scale_1, scale_2], axis=-1)
    
    # Extract rotations from PLY (wxyz format: rot_0=w, rot_1=x, rot_2=y, rot_3=z)
    rot_0 = np.array(vertex['rot_0'], dtype=np.float32)  # w
    rot_1 = np.array(vertex['rot_1'], dtype=np.float32)  # x
    rot_2 = np.array(vertex['rot_2'], dtype=np.float32)  # y
    rot_3 = np.array(vertex['rot_3'], dtype=np.float32)  # z
    # Store as wxyz (PLY format)
    rotations = np.stack([rot_0, rot_1, rot_2, rot_3], axis=-1)
    rotations = normalize_quaternion(rotations)
    
    # Extract colors (SH coefficients)
    sh_props = [p for p in vertex.data.dtype.names if p.startswith('f_dc_') or p.startswith('f_rest_')]
    sh_data = {}
    for prop in sh_props:
        sh_data[prop] = np.array(vertex[prop], dtype=np.float32)
    
    # Extract opacity
    opacity = np.array(vertex['opacity'], dtype=np.float32)
    
    # Extract normals if present
    normals = None
    if 'nx' in vertex.data.dtype.names:
        nx = np.array(vertex['nx'], dtype=np.float32)
        ny = np.array(vertex['ny'], dtype=np.float32)
        nz = np.array(vertex['nz'], dtype=np.float32)
        normals = np.stack([nx, ny, nz], axis=-1)
    
    print(f"  Loaded {len(positions)} splats")
    
    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,  # wxyz format (PLY convention)
        'opacity': opacity,
        'sh_data': sh_data,
        'normals': normals,
        'plydata': plydata  # Keep original for property order
    }


def save_ply(output_path, data):
    """Save the deformed point cloud as a PLY file."""
    print(f"Saving PLY: {output_path}")
    
    n = len(data['positions'])
    
    # Build vertex data
    vertex_data = []
    dtype_list = []
    
    # Positions
    dtype_list.extend([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # Normals
    if data['normals'] is not None:
        dtype_list.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    
    # SH coefficients (preserve original order)
    sh_props = sorted(data['sh_data'].keys())
    for prop in sh_props:
        dtype_list.append((prop, 'f4'))
    
    # Opacity
    dtype_list.append(('opacity', 'f4'))
    
    # Scales
    dtype_list.extend([('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4')])
    
    # Rotations
    dtype_list.extend([('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')])
    
    # Create structured array
    vertex_array = np.empty(n, dtype=dtype_list)
    
    # Fill in data
    vertex_array['x'] = data['positions'][:, 0]
    vertex_array['y'] = data['positions'][:, 1]
    vertex_array['z'] = data['positions'][:, 2]
    
    if data['normals'] is not None:
        vertex_array['nx'] = data['normals'][:, 0]
        vertex_array['ny'] = data['normals'][:, 1]
        vertex_array['nz'] = data['normals'][:, 2]
    
    for prop in sh_props:
        vertex_array[prop] = data['sh_data'][prop]
    
    vertex_array['opacity'] = data['opacity']
    
    vertex_array['scale_0'] = data['scales'][:, 0]
    vertex_array['scale_1'] = data['scales'][:, 1]
    vertex_array['scale_2'] = data['scales'][:, 2]
    
    # PLY uses wxyz format: rot_0=w, rot_1=x, rot_2=y, rot_3=z
    vertex_array['rot_0'] = data['rotations'][:, 0]  # w
    vertex_array['rot_1'] = data['rotations'][:, 1]  # x
    vertex_array['rot_2'] = data['rotations'][:, 2]  # y
    vertex_array['rot_3'] = data['rotations'][:, 3]  # z
    
    # Create PLY element and data
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    plydata = PlyData([vertex_element], text=False)
    plydata.write(output_path)
    
    print(f"  Saved {n} splats")


def run_deformation(ply_path, model_path, timestep, output_path, batch_size=10000):
    """Run deformation at specified timestep and save result."""
    
    # Load model
    print(f"Loading model: {model_path}")
    model = DeformNetwork(D=8, W=256, verbose=False)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded successfully!")
    
    # Load PLY
    data = load_ply(ply_path)
    positions = data['positions']
    scales = data['scales']
    rotations = data['rotations']  # wxyz format from PLY
    n = len(positions)
    
    print(f"\nRunning deformation at t={timestep}...")
    
    # Run inference in batches
    all_d_xyz = []
    all_d_rot = []
    all_d_scale = []
    
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_xyz = torch.tensor(positions[start:end], dtype=torch.float32)
            batch_t = torch.full((end - start, 1), timestep, dtype=torch.float32)
            
            d_xyz, d_rot, d_scale = model(batch_xyz, batch_t)
            
            all_d_xyz.append(d_xyz.numpy())
            all_d_rot.append(d_rot.numpy())
            all_d_scale.append(d_scale.numpy())
            
            if start % 50000 == 0:
                print(f"  Processed {start}/{n} splats...")
    
    d_xyz = np.concatenate(all_d_xyz, axis=0)
    d_rot = np.concatenate(all_d_rot, axis=0)
    d_scale = np.concatenate(all_d_scale, axis=0)
    
    print(f"  Deformation complete!")
    
    # Print statistics
    print(f"\nDeformation statistics at t={timestep}:")
    print(f"  d_xyz:   mean={np.mean(np.abs(d_xyz)):.6f}, max={np.max(np.abs(d_xyz)):.6f}")
    print(f"  d_rot:   mean={np.mean(np.abs(d_rot)):.6f}, max={np.max(np.abs(d_rot)):.6f}")
    print(f"  d_scale: mean={np.mean(np.abs(d_scale)):.6f}, max={np.max(np.abs(d_scale)):.6f}")
    
    # Apply deformations
    print("\nApplying deformations...")
    
    # Position: simple addition
    new_positions = positions + d_xyz
    
    # Rotation: add delta then normalize (matching Metal implementation)
    # Network outputs d_rot in same format as input (wxyz)
    new_rotations = rotations + d_rot
    new_rotations = normalize_quaternion(new_rotations)
    
    # Scale: add in log space (scales are already in log space in PLY)
    new_scales = scales + d_scale
    
    # Update data
    data['positions'] = new_positions
    data['rotations'] = new_rotations
    data['scales'] = new_scales
    
    # Save result
    save_ply(output_path, data)
    
    print(f"\nDone! Deformed PLY saved to: {output_path}")
    
    # Print comparison of first few splats
    print("\nFirst 5 splats comparison:")
    print("-" * 80)
    for i in range(min(5, n)):
        print(f"Splat {i}:")
        print(f"  Original pos: ({positions[i, 0]:.6f}, {positions[i, 1]:.6f}, {positions[i, 2]:.6f})")
        print(f"  Deformed pos: ({new_positions[i, 0]:.6f}, {new_positions[i, 1]:.6f}, {new_positions[i, 2]:.6f})")
        print(f"  Delta xyz:    ({d_xyz[i, 0]:.6f}, {d_xyz[i, 1]:.6f}, {d_xyz[i, 2]:.6f})")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export deformed PLY at a specific timestep")
    parser.add_argument("--ply", required=True, type=str, help="Path to input point_cloud.ply")
    parser.add_argument("--model", required=True, type=str, help="Path to deform.pth")
    parser.add_argument("--time", default=0.5, type=float, help="Timestep (0.0 to 1.0)")
    parser.add_argument("--output", default=None, type=str, help="Output PLY path")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.ply.replace('.ply', f'_deformed_{args.time}.ply')
    
    run_deformation(args.ply, args.model, args.time, args.output)
