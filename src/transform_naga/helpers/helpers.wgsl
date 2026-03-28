// Cubic Hermite spline interpolation. Matches hermite() in utilsf64.rs.
//   t  — interpolation parameter in [0, 1]
//   p0 — start value
//   p1 — end value
//   m0 — start tangent
//   m1 — end tangent
fn hermite(t: f32, p0: f32, p1: f32, m0: f32, m1: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * p0
         + (t3 - 2.0 * t2 + t) * m0
         + (-2.0 * t3 + 3.0 * t2) * p1
         + (t3 - t2) * m1;
}


// Minecraft Perlin noise — matches sample_perlin() in perlin.rs.
//
// NOTE: The per-sampler origin offsets (pns.origin_x/y/z) are NOT applied here.
// The caller must pre-apply them to `pos` before calling (or fold them into
// the origin uniform), since they cannot be stored alongside the perm table
// with the current binding layout.

fn perlin_perm_(perm: array<i32, 256>, index: i32) -> i32 {
    return perm[u32(index & 255)] & 255;
}

// Dot product with one of Minecraft's 16 gradient vectors (GRAD3 in perlin.rs).
fn perlin_grad_(hash: i32, x: f32, y: f32, z: f32) -> f32 {
    let h = u32(hash & 15);
    var gx = array<f32, 16>( 1., -1.,  1., -1.,  1., -1.,  1., -1.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  0.);
    var gy = array<f32, 16>( 1.,  1., -1., -1.,  0.,  0.,  0.,  0.,  1., -1.,  1., -1.,  1., -1.,  1., -1.);
    var gz = array<f32, 16>( 0.,  0.,  0.,  0.,  1.,  1., -1., -1.,  1.,  1., -1., -1.,  0.,  1.,  0., -1.);
    return gx[h] * x + gy[h] * y + gz[h] * z;
}

// Quintic fade curve: 6t^5 - 15t^4 + 10t^3
fn perlin_fade_(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin_lerp_(delta: f32, a: f32, b: f32) -> f32 {
    return a + delta * (b - a);
}

struct PerlinNoiseGenerator {
    perm: array<i32, 256>,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
}

// Perlin noise. Matches Minecraft's ImprovedNoise / sample_perlin_section with fade_y == yf.
// Signature matches what the SPMT code generator emits:
//   fn perlin(pos: vec3<f32>, perm: array<i32, 256>) -> f32
fn perlin(pos: vec3<f32>, generator: PerlinNoiseGenerator) -> f32 {
    let rpos3 = vec3<f32>(
        pos.x + generator.origin_x,
        pos.y + generator.origin_y,
        pos.z + generator.origin_z,
    );

    let xi = i32(floor(rpos3.x));
    let yi = i32(floor(rpos3.y));
    let zi = i32(floor(rpos3.z));

    let xf = rpos3.x - f32(xi);
    let yf = rpos3.y - f32(yi);
    let zf = rpos3.z - f32(zi);

    // Hash the lattice cube corner indices through the permutation table.
    let mi = perlin_perm_(generator.perm, xi);
    let mj = perlin_perm_(generator.perm, xi + 1);
    let k  = perlin_perm_(generator.perm, mi + yi);
    let l  = perlin_perm_(generator.perm, mi + yi + 1);
    let m  = perlin_perm_(generator.perm, mj + yi);
    let n  = perlin_perm_(generator.perm, mj + yi + 1);

    // Gradient dot products at the 8 corners of the unit cube.
    let d = perlin_grad_(perlin_perm_(generator.perm, k + zi),       xf,        yf,        zf);
    let e = perlin_grad_(perlin_perm_(generator.perm, m + zi),       xf - 1.0,  yf,        zf);
    let f = perlin_grad_(perlin_perm_(generator.perm, l + zi),       xf,        yf - 1.0,  zf);
    let g = perlin_grad_(perlin_perm_(generator.perm, n + zi),       xf - 1.0,  yf - 1.0,  zf);
    let h = perlin_grad_(perlin_perm_(generator.perm, k + zi + 1),   xf,        yf,        zf - 1.0);
    let o = perlin_grad_(perlin_perm_(generator.perm, m + zi + 1),   xf - 1.0,  yf,        zf - 1.0);
    let p = perlin_grad_(perlin_perm_(generator.perm, l + zi + 1),   xf,        yf - 1.0,  zf - 1.0);
    let q = perlin_grad_(perlin_perm_(generator.perm, n + zi + 1),   xf - 1.0,  yf - 1.0,  zf - 1.0);

    let rx = perlin_fade_(xf);
    let ry = perlin_fade_(yf); // fade_y == yf (sample_perlin passes h twice)
    let rz = perlin_fade_(zf);

    // Trilinear interpolation: lerp3(rx, ry, rz, d, e, f, g, h, o, p, q)
    return perlin_lerp_(rz,
        perlin_lerp_(ry,
            perlin_lerp_(rx, d, e),
            perlin_lerp_(rx, f, g)
        ),
        perlin_lerp_(ry,
            perlin_lerp_(rx, h, o),
            perlin_lerp_(rx, p, q)
        )
    );
}

// Maps a cave value to a scale factor. Matches scale_caves() in utilsf64.rs.
fn scale_caves(value: f32) -> f32 {
    if value < -0.75 {
        return 0.5;
    } else if value < -0.5 {
        return 0.75;
    } else if value < 0.5 {
        return 1.0;
    } else if value < 0.75 {
        return 2.0;
    }
    return 3.0;
}

// Maps a tunnel value to a scale factor. Matches scale_tunnels() in utilsf64.rs.
fn scale_tunnels(value: f32) -> f32 {
    if value < -0.5 {
        return 0.75;
    } else if value < 0.0 {
        return 1.0;
    } else if value < 0.5 {
        return 1.5;
    }
    return 2.0;
}

// Clamped linear remap from [from_y, to_y] → [from_value, to_value].
// Matches y_clamped_gradient() / clampedMap() in utilsf64.rs.
fn y_clamped_gradient(y: f32, from_y: f32, to_y: f32, from_value: f32, to_value: f32) -> f32 {
    let delta = (y - from_y) / (to_y - from_y);
    if delta <= 0.0 {
        return from_value;
    } else if delta >= 1.0 {
        return to_value;
    }
    return from_value + delta * (to_value - from_value);
}


// ==========================================
// Trilinear interpolation helpers
// Matches interpolate() / lerp() in mathf64.rs
// ==========================================

fn lerp_(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

// Trilinear interpolation of 8 corner values.
// Matches interpolate() in mathf64.rs.
// Corner naming: v{x}{y}{z} where 0 = near, 1 = far.
fn interpolate(
    v000: f32, v100: f32, v010: f32, v110: f32,
    v001: f32, v101: f32, v011: f32, v111: f32,
    fx: f32, fy: f32, fz: f32,
) -> f32 {
    let x00 = lerp_(v000, v100, fx);
    let x10 = lerp_(v010, v110, fx);
    let x01 = lerp_(v001, v101, fx);
    let x11 = lerp_(v011, v111, fx);
    let y0 = lerp_(x00, x10, fy);
    let y1 = lerp_(x01, x11, fy);
    return lerp_(y0, y1, fz);
}

// Fractional position within a 4×8×4 grid cell.
// Matches xfract4 / yfract8 / zfract4 in mathf64.rs.
fn xfract4_(pos_x: u32) -> f32 { return f32(pos_x & 3u) * 0.25; }
fn yfract8_(pos_y: u32) -> f32 { return f32(pos_y & 7u) * 0.125; }
fn zfract4_(pos_z: u32) -> f32 { return f32(pos_z & 3u) * 0.25; }

// Fractional position within a 4×16×4 grid cell.
// Matches yfract16 in mathf64.rs.
fn yfract16(pos_y: u32) -> f32 { return f32(pos_y & 15u) * 0.0625; }

// Trilinear interpolation over a 4×8×4 density grid.
// The caller is responsible for fetching the 8 surrounding corner values
// from the coarse grid; this function computes the fractional coordinates
// from `pos` (global voxel position) and interpolates.
// Matches the interpolate484 call pattern emitted by SPMT codegen.
fn interpolate484(
    v000: f32, v100: f32, v010: f32, v110: f32,
    v001: f32, v101: f32, v011: f32, v111: f32,
    xfract: f32, yfract: f32, zfract: f32,
) -> f32 {
    return interpolate(
        v000, v100, v010, v110,
        v001, v101, v011, v111,
        xfract, yfract, zfract
    );
}


// Flat 3-D → 1-D index. Matches as_index() in mathf64.rs.
// stride order: z * sy * sx + y * sx + x
fn grid_index_(gx: u32, gy: u32, gz: u32, sx: u32, sy: u32) -> u32 {
    return gz * sy * sx + gy * sx + gx;
}

// ==========================================
// Corner index helpers for 4×8×4 density grid
// Matches cornerx*y*z* in mathf64.rs (y cell size = 8, shift >> 3)
// ==========================================

fn base_grid_(pos: vec3<u32>) -> vec3<u32> {
    return vec3<u32>(pos.x >> 2u, pos.y >> 3u, pos.z >> 2u);
}

fn cornerx0y0z0_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x,     g.y,     g.z,     sx, sy);
}

fn cornerx4y0z0_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x + 1u, g.y,     g.z,     sx, sy);
}

fn cornerx0y8z0_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x,     g.y + 1u, g.z,     sx, sy);
}

fn cornerx4y8z0_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x + 1u, g.y + 1u, g.z,     sx, sy);
}

fn cornerx0y0z4_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x,     g.y,     g.z + 1u, sx, sy);
}

fn cornerx4y0z4_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x + 1u, g.y,     g.z + 1u, sx, sy);
}

fn cornerx0y8z4_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x,     g.y + 1u, g.z + 1u, sx, sy);
}

fn cornerx4y8z4_8(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_(pos);
    return grid_index_(g.x + 1u, g.y + 1u, g.z + 1u, sx, sy);
}

// ==========================================
// 4×16×4 grid helpers (y cell size = 16)
// ==========================================

fn base_grid_16_(pos: vec3<u32>) -> vec3<u32> {
    return vec3<u32>(pos.x >> 2u, pos.y >> 4u, pos.z >> 2u);
}

fn cornerx0y0z0_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x,     g.y,     g.z,     sx, sy);
}

fn cornerx4y0z0_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x + 1u, g.y,     g.z,     sx, sy);
}

fn cornerx0y16z0_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x,     g.y + 1u, g.z,     sx, sy);
}

fn cornerx4y16z0_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x + 1u, g.y + 1u, g.z,     sx, sy);
}

fn cornerx0y0z4_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x,     g.y,     g.z + 1u, sx, sy);
}

fn cornerx4y0z4_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x + 1u, g.y,     g.z + 1u, sx, sy);
}

fn cornerx0y16z4_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x,     g.y + 1u, g.z + 1u, sx, sy);
}

fn cornerx4y16z4_16(pos: vec3<u32>, sx: u32, sy: u32) -> u32 {
    let g = base_grid_16_(pos);
    return grid_index_(g.x + 1u, g.y + 1u, g.z + 1u, sx, sy);
}

// ==========================================
// Public fract helpers (called directly as extern functions)
// Aliases for the underscore-suffixed private helpers above.
// Matches xfract4 / yfract8 / zfract4 in mathf64.rs.
// ==========================================

fn xfract4(pos: vec3<u32>) -> f32 { return xfract4_(pos.x); }
fn yfract8(pos: vec3<u32>) -> f32 { return yfract8_(pos.y); }
fn zfract4(pos: vec3<u32>) -> f32 { return zfract4_(pos.z); }

// ==========================================
// Flat 2-D index helpers
// Matches flat_y_zero_index / flat_z_zero_index / biome_column_index in mathf64.rs
// ==========================================

// 3-D → 2-D index with y dimension flattened (y is ignored).
// stride order: z * size_x + x
fn flat_y_zero_index(pos: vec3<u32>, size_x: u32, size_y: u32) -> u32 {
    return pos.z * size_x + pos.x;
}

// 3-D → 2-D index with z dimension flattened (z is ignored).
// stride order: y * size_x + x
fn flat_z_zero_index(pos: vec3<u32>, size_x: u32, size_y: u32) -> u32 {
    return pos.y * size_x + pos.x;
}

// Biome column index: shifts x and z right by 2, clears y, then flat_y_zero_index with size_x=4.
// Matches biome_column_index() in mathf64.rs.
fn biome_column_index(pos: vec3<u32>) -> u32 {
    return flat_y_zero_index(vec3<u32>(pos.x >> 2u, 0u, pos.z >> 2u), 4u, 0u);
}


fn old_blended_noise(
    rpos3: vec3<f32>,
    xz_scale: f32,
    y_scale: f32,
    xz_factor: f32,
    y_factor: f32,
    smear_scale_multiplier: f32,
) -> f32 {
    return 0.0;
}
