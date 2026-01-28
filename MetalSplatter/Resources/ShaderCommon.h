#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant const int kMaxViewCount = 2;
constant static const half kBoundsRadius = 3;
constant static const half kBoundsRadiusSquared = kBoundsRadius*kBoundsRadius;

enum BufferIndex: int32_t
{
    BufferIndexUniforms = 0,
    BufferIndexSplat    = 1,
    BufferIndexSplatIndices = 2,
    BufferIndexClusterColor = 3,
    BufferIndexClusterID = 4,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;  // 64 bytes (offset 0)
    matrix_float4x4 viewMatrix;        // 64 bytes (offset 64)
    uint2 screenSize;                  // 8 bytes  (offset 128)

    /*
     The first N splats are represented as as 2N primitives and 4N vertex indices. The remained are represented
     as instanced of these first N. This allows us to limit the size of the indexed array (and associated memory),
     but also avoid the performance penalty of a very large number of instances.
     */
    uint splatCount;                   // 4 bytes  (offset 136)
    uint indexedSplatCount;            // 4 bytes  (offset 140)
    uint showClusterColors;            // 4 bytes  (offset 144)
    int selectedClusterID;             // 4 bytes  (offset 148) -1 means show all clusters
    
    uint showDepthVisualization;       // 4 bytes  (offset 152)
    uint _pad1;                        // 4 bytes  (offset 156) for float2 alignment
    float2 depthRange;                 // 8 bytes  (offset 160) min/max depth for visualization
    
    // Padding to ensure 16-byte alignment (total 176 bytes per Uniforms)
    uint2 _padding;                    // 8 bytes  (offset 168)
} Uniforms;

typedef struct
{
    Uniforms uniforms[kMaxViewCount];
} UniformsArray;

typedef struct
{
    packed_float3 position;
    packed_half4 color;
    packed_half3 covA;
    packed_half3 covB;
} Splat;

typedef struct
{
    float4 position [[position]];
    half2 relativePosition; // Ranges from -kBoundsRadius to +kBoundsRadius
    half4 color;
} FragmentIn;
