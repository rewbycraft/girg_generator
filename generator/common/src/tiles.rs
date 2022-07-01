//! Re-export of [`generator_core::tiles`] with additional iterator.

pub use generator_core::tiles::*;

/// Iterator over all tiles in a graph.
#[derive(Clone, Debug)]
pub struct TilesIterator {
    /// Total number of vertices.
    vertices: u64,
    /// Size of each tile.
    tile_size: u64,
    /// Current tile `i` position.
    i: u64,
    /// Current tile `j` position.
    j: u64,
}

impl TilesIterator {
    /// Create new iterator.
    ///
    /// # Arguments
    /// * `max` - The number of vertices.
    /// * `step` - The size of each tile.
    pub fn new(max: u64, step: u64) -> Self {
        TilesIterator {
            vertices: max,
            tile_size: step,
            i: 0u64,
            j: 0u64,
        }
    }
}

impl Iterator for TilesIterator {
    type Item = Tile;

    fn next(&mut self) -> Option<Self::Item> {
        if self.j >= self.vertices {
            None
        } else {
            let i_next = (self.i + self.tile_size).min(self.vertices);
            let j_next = (self.j + self.tile_size).min(self.vertices);

            let next = Tile((self.i, self.j), (i_next-1, j_next-1));

            self.i = i_next;
            if self.i >= self.vertices {
                self.j = j_next;
                self.i = 0;
            }

            Some(next)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(10000, 2000)]
    #[case(10000, 3000)]
    #[case(10000, 1999)]
    #[case(10000, 2001)]
    #[case(9999, 2000)]
    #[case(9999, 2001)]
    fn it_works(#[case] vertices: u64, #[case] tile_size: u64) {
        let exp = num_integer::div_ceil(vertices, tile_size).pow(2);
        let iter = TilesIterator::new(vertices, tile_size);
        let act = iter.clone().count() as u64;
        assert_eq!(act, exp, "testing amount of tiles");

        let sum_positions: usize = iter.clone().flat_map(|t: Tile| t.into_iter()).count();
        assert_eq!(sum_positions as u64, vertices * vertices);

        let max_x: u64 = iter.clone().map(|t| t.1.0).max().unwrap();
        assert_eq!(max_x, vertices-1, "testing max_x");

        let max_y: u64 = iter.map(|t| t.1.1).max().unwrap();
        assert_eq!(max_y, vertices-1, "testing max_x");
    }
}
