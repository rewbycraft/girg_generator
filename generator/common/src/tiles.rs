#[derive(Clone, Debug)]
pub struct TilesIterator {
    vertices: u64,
    tile_size: u64,
    i: u64,
    j: u64,
}

impl TilesIterator {
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
    type Item = ((u64, u64), (u64, u64));

    fn next(&mut self) -> Option<Self::Item> {
        if self.j >= self.vertices {
            None
        } else {
            let i_next = (self.i + self.tile_size).min(self.vertices);
            let j_next = (self.j + self.tile_size).min(self.vertices);

            let next = ((self.i, self.j), (i_next, j_next));

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

        let max_x: u64 = iter.clone().map(|((_, _), (x, _))| x).max().unwrap();
        assert_eq!(max_x, vertices, "testing max_x");

        let max_y: u64 = iter.map(|((_, _), (_, x))| x).max().unwrap();
        assert_eq!(max_y, vertices, "testing max_x");
    }
}
