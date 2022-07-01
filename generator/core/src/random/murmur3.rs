//! Implementation of murmur3 hash.

// Copyright (c) 2020 Stu Small
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

// From https://github.com/stusmall/murmur3/blob/master/src/murmur3_32.rs

// Modified for our purposes

const C1: u32 = 0x85eb_ca6b;
const C2: u32 = 0xc2b2_ae35;
const R1: u32 = 16;
const R2: u32 = 13;
const M: u32 = 5;
const N: u32 = 0xe654_6b64;

/// Hash three [u64] values together using the Murmur3Hash algorithm.
pub fn murmur3_32_3(s1: u64, s2: u64, s3: u64) -> u32 {
    let seed = 0u32;
    let mut processed = 0;
    let mut state = seed;

    let mut process_4 = |buffer: [u8; 4]| {
        processed += 4;
        let k = u32::from_le_bytes(buffer);
        state ^= calc_k(k);
        state = state.rotate_left(R2);
        state = (state.wrapping_mul(M)).wrapping_add(N);
    };

    let mut process_64 = |s: u64| {
        let bytes = s.to_be_bytes();
        {
            let mut buffer = [0u8; 4];
            buffer.copy_from_slice(&bytes[0..4]);
            process_4(buffer);
        }
        {
            let mut buffer = [0u8; 4];
            buffer.copy_from_slice(&bytes[4..8]);
            process_4(buffer);
        }
    };

    process_64(s1);
    process_64(s2);
    process_64(s3);

    finish(state, processed)
}

/// Hash two [u64] values together using the Murmur3Hash algorithm.
pub fn murmur3_32_2(s1: u64, s2: u64) -> u32 {
    let seed = 0u32;
    let mut processed = 0;
    let mut state = seed;

    let mut process_4 = |buffer: [u8; 4]| {
        processed += 4;
        let k = u32::from_le_bytes(buffer);
        state ^= calc_k(k);
        state = state.rotate_left(R2);
        state = (state.wrapping_mul(M)).wrapping_add(N);
    };

    let mut process_64 = |s: u64| {
        let bytes = s.to_be_bytes();
        {
            let mut buffer = [0u8; 4];
            buffer.copy_from_slice(&bytes[0..4]);
            process_4(buffer);
        }
        {
            let mut buffer = [0u8; 4];
            buffer.copy_from_slice(&bytes[4..8]);
            process_4(buffer);
        }
    };

    process_64(s1);
    process_64(s2);

    finish(state, processed)
}

fn finish(state: u32, processed: u32) -> u32 {
    let mut hash = state;
    hash ^= processed as u32;
    hash ^= hash.wrapping_shr(R1);
    hash = hash.wrapping_mul(C1);
    hash ^= hash.wrapping_shr(R2);
    hash = hash.wrapping_mul(C2);
    hash ^= hash.wrapping_shr(R1);
    hash
}

fn calc_k(k: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    const R1: u32 = 15;
    k.wrapping_mul(C1).rotate_left(R1).wrapping_mul(C2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mur3::murmurhash3_x86_32;
    use rand::Rng;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;

    #[test]
    fn test_equality_3() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _ in 0..10240 {
            let s1: u64 = rng.gen();
            let s2: u64 = rng.gen();
            let s3: u64 = rng.gen();

            let expected = murmurhash3_x86_32(
                &[s1.to_be_bytes(), s2.to_be_bytes(), s3.to_be_bytes()].concat(),
                0,
            );
            let actual = murmur3_32_3(s1, s2, s3);
            assert_eq!(actual, expected, "hashing [{}, {}, {}]", s1, s2, s3);
        }
    }

    #[test]
    fn test_equality_2() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _ in 0..10240 {
            let s1: u64 = rng.gen();
            let s2: u64 = rng.gen();

            let expected = murmurhash3_x86_32(&[s1.to_be_bytes(), s2.to_be_bytes()].concat(), 0);
            let actual = murmur3_32_2(s1, s2);
            assert_eq!(actual, expected, "hashing [{}, {}]", s1, s2);
        }
    }
}
