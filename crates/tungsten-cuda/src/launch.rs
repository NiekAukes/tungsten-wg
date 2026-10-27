/*
Block and grid shape of an instance's kernel. Deterministic, so the device side
(`__launch_bounds__`) and the host side (`<<<grid, block>>>`) compute the same shape. Keep this and
the kernel body signature stable: wave fusion (SPEC.md §9.4) builds on both.
*/

/// Block and grid dimensions of one launch, as `(x, y, z)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchShape {
    pub block: (u32, u32, u32),
    pub grid: (u32, u32, u32),
}

impl LaunchShape {
    pub fn threads_per_block(&self) -> u32 {
        self.block.0 * self.block.1 * self.block.2
    }
}

/// The shape for `dimensions` with at most `block_size` threads per block (a power of two):
/// x gets as many threads as it needs first, then y, then z.
pub fn launch_shape(dimensions: (i32, i32, i32), block_size: u32) -> LaunchShape {
    let (dx, dy, dz) = dimensions;
    assert!(dx > 0 && dy > 0 && dz > 0, "instance dimensions must be positive, got {:?}", dimensions);
    let elements = dx as i64 * dy as i64 * dz as i64;
    assert!(elements <= i32::MAX as i64, "instance {:?} has more than i32::MAX elements", dimensions);
    let (dx, dy, dz) = (dx as u32, dy as u32, dz as u32);

    let bx = dx.next_power_of_two().min(block_size);
    let by = dy.next_power_of_two().min(block_size / bx);
    let bz = dz.next_power_of_two().min(block_size / (bx * by));
    let grid = (dx.div_ceil(bx), dy.div_ceil(by), dz.div_ceil(bz));
    assert!(
        grid.1 <= 65535 && grid.2 <= 65535,
        "grid {:?} for instance {:?} exceeds the y/z limit of 65535",
        grid,
        dimensions
    );
    LaunchShape { block: (bx, by, bz), grid }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let s = launch_shape((16, 16, 16), 128);
        assert_eq!(s.block, (16, 8, 1));
        assert_eq!(s.grid, (1, 2, 16));
        let s = launch_shape((5, 49, 5), 128);
        assert_eq!(s.block, (8, 16, 1));
        assert_eq!(s.grid, (1, 4, 5));
        let s = launch_shape((1, 1, 1), 128);
        assert_eq!(s.block, (1, 1, 1));
        assert_eq!(s.grid, (1, 1, 1));
        let s = launch_shape((1000, 3, 2), 256);
        assert_eq!(s.block, (256, 1, 1));
        assert_eq!(s.grid, (4, 3, 2));
        assert!(s.threads_per_block() <= 256);
    }
}
