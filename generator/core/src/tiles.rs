//! Module to contain definition of a [`Tile`] and an [`Edge`].

/// A directed edge between two nodes.
pub type Edge = (u64, u64);

/// A tile.
///
/// Defined as the pair of edges `(p1, p2)` in the top left and bottom right of the tile.
/// That is, `p1.0 < p2.0` and `p1.1 < p2.1`.
#[derive(Clone, Copy, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct Tile(pub Edge, pub Edge);

impl IntoIterator for Tile {
    type Item = Edge;
    type IntoIter = TileIterator;

    fn into_iter(self) -> Self::IntoIter {
        TileIterator::from(self)
    }
}

/// Iterator over edges in a tile.
#[derive(Clone)]
pub struct TileIterator {
    /// Current `i` position.
    i: u64,
    /// Current `j` position.
    j: u64,
    /// Tile we're iterating over.
    t: Tile,
}

impl From<Tile> for TileIterator {
    fn from(t: Tile) -> Self {
        Self {
            i: t.0.0,
            j: t.0.1,
            t,
        }
    }
}

impl TileIterator {
    /// Advance this iterator to the specified position in the tile.
    pub fn skip_to(&mut self, i: u64, j: u64) -> Self {
        if i <= self.t.1.0 && j <= self.t.1.1 {
            self.i = i;
            self.j = j;
            self.clone()
        } else {
            panic!("Invalid position! {},{} not in tile {:?}", i, j, self.t);
        }
    }
}

impl Iterator for TileIterator {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i <= self.t.1.0 && self.j <= self.t.1.1 {
            let c_i = self.i;
            let c_j = self.j;

            self.i += 1;
            if self.i > self.t.1.0 {
                self.i = self.t.0.0;
                self.j += 1;
            }

            Some((c_i, c_j))
        } else {
            None
        }
    }
}