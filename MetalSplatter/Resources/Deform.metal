#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

struct CanonicalSplat {
    packed_float3 position;
    packed_half4 color;
    // Quaternion stored in xyzw format (x, y, z, w) - matches simd_quatf convention
    // Note: PLY files use wxyz (rot_0=w), Swift converts to xyzw when loading
    float rotationX;  // x (imag.x)
    float rotationY;  // y (imag.y)
    float rotationZ;  // z (imag.z)
    float rotationW;  // w (real/scalar)
    packed_float3 scale;
    // Pre-computed covariance from Swift (same computation as Splat init)
    // This ensures t=0 output matches direct PLY loading exactly
    packed_half3 covA;  // (cov[0,0], cov[0,1], cov[0,2])
    packed_half3 covB;  // (cov[1,1], cov[1,2], cov[2,2])
};

// Covariance computation matching TRASE/3DGS convention
// Reference: https://github.com/yunjinli/TRASE/blob/master/utils/general_utils.py#L97
// Quaternion order: (x, y, z, w) where w is scalar part
// Metal float3x3 is column-major, so we pass columns
void compute_cov(float4 rot, float3 scale, thread packed_half3 &covA, thread packed_half3 &covB) {
    float x = rot.x, y = rot.y, z = rot.z, w = rot.w;
    
    // Build rotation matrix (column-major for Metal)
    // Column 0: [R00, R10, R20]
    // Column 1: [R01, R11, R21]
    // Column 2: [R02, R12, R22]
    float3x3 R = float3x3(
        float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),       2.0 * (x * z - w * y)),       // column 0
        float3(2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),       // column 1
        float3(2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y))  // column 2
    );
    
    // Diagonal scale matrix
    float3x3 S = float3x3(
        float3(scale.x, 0, 0),
        float3(0, scale.y, 0),
        float3(0, 0, scale.z)
    );
    
    float3x3 M = R * S;
    float3x3 Sigma = M * transpose(M);
    
    // Extract upper triangle of covariance matrix
    // Metal indexing: Sigma[col][row], Swift indexing: cov3D[row, col]
    // covA = (Σ00, Σ01, Σ02), covB = (Σ11, Σ12, Σ22)
    covA = packed_half3(half(Sigma[0][0]), half(Sigma[1][0]), half(Sigma[2][0]));
    covB = packed_half3(half(Sigma[1][1]), half(Sigma[2][1]), half(Sigma[2][2]));
}

// Extract xyz and t from the canonical Gaussians.
kernel void extract_graph_inputs(
    device const CanonicalSplat* inSplats [[ buffer(0) ]],
    device float* outXYZ                [[ buffer(1) ]],
    device float* outT                  [[ buffer(2) ]],
    constant float& time                [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    CanonicalSplat s = inSplats[id];
    outXYZ[id * 3 + 0] = s.position.x;
    outXYZ[id * 3 + 1] = s.position.y;
    outXYZ[id * 3 + 2] = s.position.z;
    outT[id] = time;
}

kernel void fill_time(
    device float* outT       [[ buffer(0) ]],
    constant float& time     [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    outT[id] = time;
}

// Apply d_xyz, d_rotation, d_scaling to the canonical Gaussians.
kernel void apply_graph_outputs(
    device const CanonicalSplat* inSplats [[ buffer(0) ]],
    device const float* dXYZ              [[ buffer(1) ]],
    device const float* dRot              [[ buffer(2) ]],
    device const float* dScale            [[ buffer(3) ]],
    device Splat* outSplats               [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    CanonicalSplat input = inSplats[id];
    
    float3 d_xyz = float3(dXYZ[id*3+0], dXYZ[id*3+1], dXYZ[id*3+2]);
    
    // Network outputs rotation delta in wxyz format (w, x, y, z)
    // But CanonicalSplat stores rotation in xyzw format (x, y, z, w)
    // So we need to reorder: wxyz -> xyzw
    float4 d_rot_wxyz = float4(dRot[id*4+0], dRot[id*4+1], dRot[id*4+2], dRot[id*4+3]);
    float4 d_rotation = float4(d_rot_wxyz.y, d_rot_wxyz.z, d_rot_wxyz.w, d_rot_wxyz.x);  // xyzw
    
    float3 d_scaling = float3(dScale[id*3+0], dScale[id*3+1], dScale[id*3+2]);
    
    // Apply position deformation
    float3 new_pos = input.position + d_xyz;
    
    Splat out;
    out.position = packed_float3(new_pos);
    out.color = input.color;
    
    // Check if rotation/scale deltas are effectively zero
    // If so, use the pre-computed covariance from Swift (matches direct PLY loading exactly)
    float rot_delta_sq = dot(d_rotation, d_rotation);
    float scale_delta_sq = dot(d_scaling, d_scaling);
    
    if (rot_delta_sq < 1e-12 && scale_delta_sq < 1e-12) {
        // No rotation/scale deformation - use pre-computed covariance
        out.covA = input.covA;
        out.covB = input.covB;
    } else {
        // Apply rotation/scale deformation and recompute covariance
        float4 rot = float4(input.rotationX, input.rotationY, input.rotationZ, input.rotationW);
        // Add delta to rotation, then normalize (matching TRASE convention)
        float4 new_rot = normalize(rot + d_rotation);
        
        // Delta scaling is added in log-space (matching TRASE), then convert to linear for covariance
        float3 new_log_scale = input.scale + d_scaling;
        float3 linear_scale = exp(new_log_scale);
        
        // compute_cov expects LINEAR scale
        compute_cov(new_rot, linear_scale, out.covA, out.covB);
    }
    
    outSplats[id] = out;
}

