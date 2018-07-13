#![feature(test)]

extern crate rayon;
extern crate test;

use std::ops::*;

use rayon::prelude::*;

#[cfg(test)]
use test::Bencher;

const NB_MODULI: usize = 5;
const MODULI: [i64; NB_MODULI] = [89702869, 78489023, 69973811, 70736797, 79637461];

#[derive(Clone, Debug, PartialEq)]
pub struct CrtScalar([i64; NB_MODULI]);

impl From<i64> for CrtScalar {
    fn from(x: i64) -> Self {
        let mut backing = [0; NB_MODULI];
        backing.iter_mut().enumerate()
            .for_each(|(i, zi)| {
                let mi = MODULI[i];
                *zi = x % mi;
            });
        CrtScalar(backing)
    }
}

impl Index<usize> for CrtScalar {
    type Output = i64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

#[test]
fn test_from() {
    let x = CrtScalar::from(123456789);
    assert_eq!(x.0, [33753920, 44967766, 53482978, 52719992, 43819328])
}

#[bench]
fn bench_from(b: &mut Bencher) {
    b.iter(|| {
        let _ = CrtScalar::from(123456789);
    })
}

// impl Add<CrtScalar> for CrtScalar {
//     type Output = CrtScalar;
//     fn add(self, other: CrtScalar) -> Self::Output {
//         let mut z = [0; NB_MODULI];
//         z.iter_mut().enumerate()
//             .for_each(|(i, zi)| {
//                 let xi = self[i];
//                 let yi = other[i];
//                 let mi = MODULI[i];
//                 *zi = (xi + yi) % mi
//             });
//         CrtScalar(z)
//     }
// }

impl<'s, 'o> Add<&'o CrtScalar> for &'s CrtScalar {
    type Output = CrtScalar;
    fn add(self, other: &'o CrtScalar) -> Self::Output {
        let mut backing = [0; NB_MODULI];
        backing.iter_mut().enumerate()
            .for_each(|(i, zi)| {
                let xi = self[i];
                let yi = other[i];
                let mi = MODULI[i];
                *zi = (xi + yi) % mi
            });
        CrtScalar(backing)
    }
}

#[test]
fn test_add() {
    let x = CrtScalar::from(1234567);
    let y = CrtScalar::from(7654321);
    let z = &x + &y;
    assert_eq!(z, CrtScalar::from(1234567+7654321))
}

// #[bench]
// fn bench_add_crt(b: &mut Bencher) {
//     let x = CrtScalar::from(1234567);
//     let y = CrtScalar::from(7654321);
//     b.iter(|| {
//         let _z = &x + &y;
//     })
// }

// #[bench]
// fn bench_add_native(b: &mut Bencher) {
//     let x = 1234567;
//     let y = 7654321;
//     b.iter(|| {
//         let _z = &x + &y;
//     })
// }

// impl<'s, 'o> Sub<&'o CrtScalar> for &'s CrtScalar {
//     type Output = CrtScalar;
//     fn sub(self, other: &'o CrtScalar) -> Self::Output {
//         let mut backing = [0; NB_MODULI];
//         backing.iter_mut().enumerate()
//             .for_each(|(i, zi)| {
//                 let xi = self[i];
//                 let yi = other[i];
//                 let mi = MODULI[i];
//                 *zi = (xi - yi) % mi;
//             });
//         CrtScalar(backing)
//     }
// }

// #[test]
// fn test_sub() {
//     let x = CrtScalar::from(1234567);
//     let y = CrtScalar::from(7654321);
//     let z = &x - &y;
//     assert_eq!(z, CrtScalar::from(1234567-7654321))
// }

// impl<'s, 'o> Mul<&'o CrtScalar> for &'s CrtScalar {
//     type Output = CrtScalar;
//     fn mul(self, other: &'o CrtScalar) -> Self::Output {
//         let mut backing = [0; NB_MODULI];
//         backing.iter_mut().enumerate()
//             .for_each(|(i, zi)| {
//                 let xi = self[i];
//                 let yi = other[i];
//                 let mi = MODULI[i];
//                 *zi = (xi * yi) % mi
//             });
//         CrtScalar(backing)
//     }
// }

// #[test]
// fn test_mul() {
//     let x = CrtScalar::from(1234567);
//     let y = CrtScalar::from(7654321);
//     let z = &x * &y;
//     assert_eq!(z, CrtScalar::from(1234567 * 7654321))
// }
