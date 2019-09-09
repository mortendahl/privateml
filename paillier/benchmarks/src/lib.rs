#![feature(test)]

extern crate gmp;
extern crate rayon;
extern crate test;

use gmp::mpz::Mpz as BigInt;
use gmp::sign::Sign;
use test::Bencher;


pub struct TestValues {
    pub p: BigInt,
    pub q: BigInt,
    pub x: BigInt,
    pub r: BigInt,
    pub c: BigInt,
}

impl TestValues {    
    pub fn parse() -> Self {
        let P: &'static str = "148677972634832330983979593310074301486537017973460461278300587514468301043894574906886127642530475786889672304776052879927627556769456140664043088700743909632312483413393134504352834240399191134336344285483935856491230340093391784574980688823380828143810804684752914935441384845195613674104960646037368551517";
        let Q: &'static str = "158741574437007245654463598139927898730476924736461654463975966787719309357536545869203069369466212089132653564188443272208127277664424448947476335413293018778018615899291704693105620242763173357203898195318179150836424196645745308205164116144020613415407736216097185962171301808761138424668335445923774195463";
        let X: &'static str = "8116954461269652085230775933492366253929619979964246027246617328236243795946267984122836662596238827711003162168747438362516197517513090468979133736169125128476037082825330864610731186580002727070392849209478375921588198871833235568390694413294064765159446378195902634553122666031519508653458373801364626796907156918652837961453184912515251722492496853769056007056222605281803303377223245158930080581216344814858180859850388248516110876304421350734473865383370328091654104048265291335553686536171725033973437997155180998731315175192344098133206744942235940959001435284014629247610235987290001278275024859859011530610";
        let R: &'static str = "23469954723780858594246099985284351679616402035799877801904657600748845577732527349622767465073206402501465434997097886062288454586744002070231423281154066427914375124691393663408153322045175238293238130584029098551997891083995941940072522297040482563102994301991914687415608932934232322244422743863612925485379738169539911563673020855346364903207395531099432657381392680821739845416005854261986810418869105342379811199853817109183567590778515098199155952701550697969915592026323126896564744857660315540347612223681727710811401589383421009333736638539423942798822879086915723828754886239159868603235567977031107975333";
        let C: &'static str = "257203447105391916804089534214900683312963097096490235488958404866382741464385583184601128552670618981990000105037083690090594298564475202598013462067431119022723496775733949006049841771523035314712582893707414778394151736782745046995179058013948782711907055567514176057362937274003340106139237726460592991291649036854647465105678112034545238612525869223794770077034914371840530192251732830868984008901241006294086749915286415154943481920642073417739888367146198161135533211217852125267939460775999200943311678088646957279291231065720239752089787957176968565441059132771040186019584522752156210356879184733001364159458400826829676108356363700260706055423980900810086835364443358559049002895766426201073894808737170968355348275995156366869463099375320511865686019912874183927654191322970227080404808169296681251742918581398335645810766224246925708682914546637276241236065831895597376590899342810804823166192194221790279822050700148335320259676027171464292155454236417540711787990414904918789189297629562318139402290148344330957118511748943890681355239458737273975338090521441500322894883091962991413705362193011723285869089808996484682861997586072545454669866370589115975111183983919754128226878169359264094995923843080089523149217085";

        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let r: BigInt = str::parse(R).unwrap();
        let c: BigInt = str::parse(C).unwrap();

        TestValues { p, q, x, r, c }
    }
}

pub fn inv(a: &BigInt, modulus: &BigInt) -> BigInt {
    a.invert(modulus).unwrap()
}

pub fn pow(base: &BigInt, exponent: &BigInt, modulus: &BigInt) -> BigInt {
    base.powm(exponent, modulus)
}

pub fn l(u: &BigInt, n: &BigInt) -> BigInt {
    (u - 1) / n
}

pub fn h(m: &BigInt, mm: &BigInt, n: &BigInt) -> BigInt {
    // compute g^{p-1} mod p^2
    let gm = (1 - n) % mm;
    // compute L_p(.)
    let lm = l(&gm, m);
    // compute L_p(.)^{-1}
    inv(&lm, m)
}

pub struct CrtParams {
    m1: BigInt,
    m2: BigInt,
    m1_inv: BigInt,
}

impl CrtParams {
    pub fn new(m1: &BigInt, m2: &BigInt) -> CrtParams {
        let m1_inv = inv(m1, m2);
        let m1 = m1.clone();
        let m2 = m2.clone();
        CrtParams { m1, m2, m1_inv }
    }
}

pub fn crt(params: &CrtParams, x1: &BigInt, x2: &BigInt) -> BigInt {
    let mut diff = x2 - x1;
    if diff.sign() == Sign::Negative {
        diff = (diff % &params.m2) + &params.m2;
    }
    let u = (diff * &params.m1_inv) % &params.m2;
    let x = x1 + (u * &params.m1);
    x
}

pub struct Keypair {
    p: BigInt,
    q: BigInt,
}

mod plain {

    use ::*;

    struct EncryptionKey {
        n: BigInt,
        nn: BigInt,
        g: BigInt,
    }

    impl EncryptionKey {
        fn from(keypair: &Keypair) -> EncryptionKey {
            let n = &keypair.p * &keypair.q;
            let nn = &n * &n;
            let g = 1 + &n;

            EncryptionKey { n, nn, g }
        }
    }

    #[bench]
    fn bench_encryption_key(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        
        let keypair = Keypair { p, q };

        b.iter(|| {
            let _ = EncryptionKey::from(&keypair);
        });
    }

    fn encrypt(ek: &EncryptionKey, x: &BigInt, r: &BigInt) -> BigInt {
        let gx = pow(&ek.g, x, &ek.nn);
        let rm = pow(r, &ek.n, &ek.nn);
        let c = (gx * rm) % &ek.nn;
        c
    }

    #[test]
    fn test_encrypt() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let ek = EncryptionKey::from(&keypair);

        assert_eq!(encrypt(&ek, &x, &r), c);
    }

    #[bench]
    fn bench_encrypt(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;

        let keypair = Keypair { p, q };
        let ek = EncryptionKey::from(&keypair);

        b.iter(|| {
            let _ = encrypt(&ek, &x, &r);
        });
    }

    struct DecryptionKey {
        n: BigInt,
        nn: BigInt,
        g: BigInt,
        d1: BigInt,
        d2: BigInt,
        e: BigInt,
    }

    impl DecryptionKey {
        fn from(keypair: &Keypair) -> DecryptionKey {
            let n = &keypair.p * &keypair.q;
            let nn = &n * &n;
            let g = 1 + &n;

            let order_of_n = (&keypair.p - 1) * (&keypair.q - 1);
            let e = inv(&n, &order_of_n);
            let d2 = inv(&order_of_n, &n);
            let d1 = order_of_n;

            DecryptionKey { n, nn, g, d1, d2, e }
        }
    }

    #[bench]
    fn bench_decryption_key(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        
        let keypair = Keypair { p, q };

        b.iter(|| {
            let _ = DecryptionKey::from(&keypair);
        });
    }

    fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let gxd = pow(c, &dk.d1, &dk.nn);
        let xd = l(&gxd, &dk.n);
        let x = (xd * &dk.d2) % &dk.n;
        x
    }

    #[test]
    fn test_decrypt() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        assert_eq!(decrypt(&dk, &c), x);
    }

    #[bench]
    fn bench_decrypt(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        b.iter(|| {
            let _ = decrypt(&dk, &c);
        });
    }

    fn extract(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let x = decrypt(dk, c);
        let gx = pow(&dk.g, &x, &dk.nn);
        let gx_inv = inv(&gx, &dk.nn);
        let rn = (c * gx_inv) % &dk.nn;
        let r = pow(&rn, &dk.e, &dk.n);
        r
    }

    #[test]
    fn test_extract() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let r = test_values.r;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        assert_eq!(extract(&dk, &c), r);
    }

    #[bench]
    fn bench_extract(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        b.iter(|| {
            let _ = extract(&dk, &c);
        });
    }

}

mod specialized {

    use ::*;

    pub struct EncryptionKey {
        pub n: BigInt,
        pub nn: BigInt,
    }

    impl EncryptionKey {
        pub fn from(keypair: &Keypair) -> EncryptionKey {
            let n = &keypair.p * &keypair.q;
            let nn = &n * &n;
            
            EncryptionKey { n, nn }
        }
    }

    #[bench]
    fn bench_encryption_key(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        
        let keypair = Keypair { p, q };

        b.iter(|| {
            let _ = EncryptionKey::from(&keypair);
        });
    }

    fn encrypt(ek: &EncryptionKey, x: &BigInt, r: &BigInt) -> BigInt {
        let gx = 1 + x * &ek.n;
        let rn = pow(r, &ek.n, &ek.nn);
        let c = (gx * rn) % &ek.nn;
        c
    }

    #[test]
    fn test_encrypt() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let ek = EncryptionKey::from(&keypair);

        assert_eq!(encrypt(&ek, &x, &r), c);
    }

    #[bench]
    fn bench_encrypt(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;

        let keypair = Keypair { p, q };
        let ek = EncryptionKey::from(&keypair);

        b.iter(|| {
            let _ = encrypt(&ek, &x, &r);
        });
    }

    struct DecryptionKey {
        n: BigInt,
        nn: BigInt,
        d1: BigInt,
        d2: BigInt,
        e: BigInt,
    }

    impl DecryptionKey {
        fn from(keypair: &Keypair) -> DecryptionKey {
            let n = &keypair.p * &keypair.q;
            let nn = &n * &n;

            let order_of_n = (&keypair.p - 1) * (&keypair.q - 1);
            let e = inv(&n, &order_of_n);
            let d2 = inv(&order_of_n, &n);
            let d1 = order_of_n;

            DecryptionKey { n, nn, d1, d2, e }
        }
    }

    #[bench]
    fn bench_decryption_key(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        
        let keypair = Keypair { p, q };

        b.iter(|| {
            let _ = DecryptionKey::from(&keypair);
        });
    }

    fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let gxd = pow(c, &dk.d1, &dk.nn);
        let xd = l(&gxd, &dk.n);
        let x = (xd * &dk.d2) % &dk.n;
        x
    }

    #[test]
    fn test_decrypt() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        assert_eq!(decrypt(&dk, &c), x);
    }

    #[bench]
    fn bench_decrypt(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        b.iter(|| {
            let _ = decrypt(&dk, &c);
        });
    }

    fn extract(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let rn = c % &dk.n;
        let r = pow(&rn, &dk.e, &dk.n);
        r
    }

    #[test]
    fn test_extract() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let r = test_values.r;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        assert_eq!(extract(&dk, &c), r);
    }

    #[bench]
    fn bench_extract(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        b.iter(|| {
            let _ = extract(&dk, &c);
        });
    }

}

mod precomputed_randomness {

    use ::*;

    use ::specialized::EncryptionKey;

    fn encrypt(ek: &EncryptionKey, x: &BigInt, rn: &BigInt) -> BigInt {
        let gx = 1 + (x * &ek.n);
        let c = (gx * rn) % &ek.nn;
        c
    }

    #[test]
    fn test_encrypt() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let ek = EncryptionKey::from(&keypair);
        let rn = pow(&r, &ek.n, &ek.nn);

        assert_eq!(encrypt(&ek, &x, &rn), c);
    }

    #[bench]
    fn bench_encrypt(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;

        let keypair = Keypair { p, q };
        let ek = EncryptionKey::from(&keypair);
        let rn = pow(&r, &ek.n, &ek.nn);

        b.iter(|| {
            let _ = encrypt(&ek, &x, &rn);
        });
    }
}

mod crt {

    use ::*;

    struct DecryptionKey {
        n: BigInt,
        p: BigInt, pp: BigInt, d1p: BigInt, d2p: BigInt, ep: BigInt,
        q: BigInt, qq: BigInt, d1q: BigInt, d2q: BigInt, eq: BigInt,
        n_crt: CrtParams,
        nn_crt: CrtParams,
    }

    impl DecryptionKey {
        fn from(keypair: &Keypair) -> DecryptionKey {
            let p = keypair.p.clone();
            let q = keypair.q.clone();

            let n = &p * &q;
            let pp = &p * &p;
            let qq = &q * &q;

            let order_of_p = &p - 1;
            let ep = inv(&n, &order_of_p);
            let d2p = h(&p, &pp, &n);
            let d1p = order_of_p;
            
            let order_of_q = &q - 1;
            let eq = inv(&n, &order_of_q);
            let d2q = h(&q, &qq, &n);
            let d1q = order_of_q;

            let n_crt = CrtParams::new(&p, &q);
            let nn_crt = CrtParams::new(&pp, &qq);

            DecryptionKey {
                n,
                p, pp, d1p, d2p, ep,
                q, qq, d1q, d2q, eq,
                n_crt,
                nn_crt,
            }
        }
    }

    #[bench]
    fn bench_decryption_key(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        
        let keypair = Keypair { p, q };

        b.iter(|| {
            let _ = DecryptionKey::from(&keypair);
        });
    }

    fn encrypt(dk: &DecryptionKey, x: &BigInt, r: &BigInt) -> BigInt {
        let cp = encrypt_component(x, r, &dk.n, &dk.p, &dk.pp);
        let cq = encrypt_component(x, r, &dk.n, &dk.q, &dk.qq);
        let c = crt(&dk.nn_crt, &cp, &cq);
        c
    }

    fn encrypt_component(x: &BigInt, r: &BigInt, n: &BigInt, m: &BigInt, mm: &BigInt) -> BigInt {
        let xm = x % m;
        let rm = r % mm;
        let gx = (1 + xm * n) % mm;
        let rn = pow(&rm, n, mm);
        let cm = (gx * rn) % mm;
        cm
    }

    #[test]
    fn test_encrypt() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        assert_eq!(encrypt(&dk, &x, &r), c);
    }

    #[bench]
    fn bench_encrypt(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let r = test_values.r;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        b.iter(|| {
            let _ = encrypt(&dk, &x, &r);
        });
    }

    fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let xp = decrypt_component(c, &dk.p, &dk.pp, &dk.d1p, &dk.d2p);
        let xq = decrypt_component(c, &dk.q, &dk.qq, &dk.d1q, &dk.d2q);
        let x = crt(&dk.n_crt, &xp, &xq);
        x
    }

    fn decrypt_component(c: &BigInt, m: &BigInt, mm: &BigInt, d1: &BigInt, d2: &BigInt) -> BigInt {
        let cm = c % mm;
        let dm = pow(&cm, d1, mm);
        let lm = l(&dm, m);
        let xm = (lm * d2) % m;
        xm
    }

    #[test]
    fn test_decrypt() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let x = test_values.x;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        assert_eq!(decrypt(&dk, &c), x);
    }

    #[bench]
    fn bench_decrypt(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        b.iter(|| {
            let _ = decrypt(&dk, &c);
        });
    }

    fn extract(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let rp = extract_component(c, &dk.p, &dk.ep);
        let rq = extract_component(c, &dk.q, &dk.eq);
        let r = crt(&dk.n_crt, &rp, &rq);
        r
    }

    fn extract_component(c: &BigInt, m: &BigInt, e: &BigInt) -> BigInt {
        let rm = c % m;
        let r = pow(&rm, e, m);
        r
    }

    #[test]
    fn test_extract() {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let r = test_values.r;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        assert_eq!(extract(&dk, &c), r);
    }

    #[bench]
    fn bench_extract(b: &mut Bencher) {
        let test_values = TestValues::parse();
        let p = test_values.p;
        let q = test_values.q;
        let c = test_values.c;

        let keypair = Keypair { p, q };
        let dk = DecryptionKey::from(&keypair);

        b.iter(|| {
            let _ = extract(&dk, &c);
        });
    }

    mod parallel {

        use ::*;
        use ::crt::{DecryptionKey, encrypt_component, decrypt_component, extract_component};

        use rayon::join;

        fn encrypt(dk: &DecryptionKey, x: &BigInt, r: &BigInt) -> BigInt {
            let (cp, cq) = join(
                || encrypt_component(x, r, &dk.n, &dk.p, &dk.pp),
                || encrypt_component(x, r, &dk.n, &dk.p, &dk.qq),
            );
            crt(&dk.nn_crt, &cp, &cq)
        }

        #[bench]
        fn bench_encrypt(b: &mut Bencher) {
            let test_values = TestValues::parse();
            let p = test_values.p;
            let q = test_values.q;
            let x = test_values.x;
            let r = test_values.r;

            let keypair = Keypair { p, q };
            let dk = DecryptionKey::from(&keypair);

            b.iter(|| {
                let _ = encrypt(&dk, &x, &r);
            });
        }

        fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
            let (mp, mq) = join(
                || decrypt_component(c, &dk.p, &dk.pp, &dk.d1p, &dk.d2p),
                || decrypt_component(c, &dk.q, &dk.qq, &dk.d1q, &dk.d2q),
            );
            crt(&dk.n_crt, &mp, &mq)
        }

        #[bench]
        fn bench_decrypt(b: &mut Bencher) {
            let test_values = TestValues::parse();
            let p = test_values.p;
            let q = test_values.q;
            let c = test_values.c;

            let keypair = Keypair { p, q };
            let dk = DecryptionKey::from(&keypair);

            b.iter(|| {
                let _ = decrypt(&dk, &c);
            });
        }

        fn extract(dk: &DecryptionKey, c: &BigInt) -> BigInt {
            let (rp, rq) = join(
                || extract_component(c, &dk.p, &dk.ep),
                || extract_component(c, &dk.q, &dk.eq),
            );
            crt(&dk.n_crt, &rp, &rq)
        }

        #[bench]
        fn bench_extract(b: &mut Bencher) {
            let test_values = TestValues::parse();
            let p = test_values.p;
            let q = test_values.q;
            let c = test_values.c;

            let keypair = Keypair { p, q };
            let dk = DecryptionKey::from(&keypair);

            b.iter(|| {
                let _ = extract(&dk, &c);
            });
        }
    }
}

mod micro {

    use super::{BigInt, pow};
    use super::Bencher;

    mod decrypt_pow {

        use super::*;

        fn bench_core(b: &mut Bencher, x: &BigInt) {
            let xx = x * x;

            b.iter(|| {
                let _ = pow(&xx, x, &xx);
            });
        }

        #[bench]
        fn bench_128(b: &mut Bencher) {
            let m: BigInt = str::parse("319716896562200682735840267002423031680").unwrap();
            assert_eq!(m.bit_length(), 128);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_256(b: &mut Bencher) {
            let m: BigInt = str::parse("102218893947364930742836883904296756780388077228204429076727188202522283622400").unwrap();
            assert_eq!(m.bit_length(), 256);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_384(b: &mut Bencher) {
            let m: BigInt = str::parse("32681107542872235002065797137621157089017715588428732619515978354293601562205469617383180858688576929923020357632000").unwrap();
            assert_eq!(m.bit_length(), 384);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_512(b: &mut Bencher) {
            let m: BigInt = str::parse("10448702279822638872609219905913590437794110190935540593362910398709875330978423670349199641513167313498410151217989846716961919585686986031827265781760000").unwrap();
            assert_eq!(m.bit_length(), 512);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_640(b: &mut Bencher) {
            let m: BigInt = str::parse("3340626666007285086299377497934236566883656215523108590286279236727706013346528169447414308941117917097915023508291633456713755258992064798326663930807067050255512952769077083196984446156800000").unwrap();
            assert_eq!(m.bit_length(), 640);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_768(b: &mut Bencher) {
            let m: BigInt = str::parse("1068054790228780493573578610727934338898260758298100943236502322499743433549133047587823771294519457781057638346141506666421153185563563771901210431094177622622400432159801647571060469187655951488005304889415124859105180647424000000").unwrap();
            assert_eq!(m.bit_length(), 768);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_896(b: &mut Bencher) {
            let m: BigInt = str::parse("341475162890337961536597552564452739524495149868723488873405081024621230153007619235774825850011916457863131785046125078073961041192543286820920095514268251942221126819184200855220635510699833506709726553463827218962888553657369089536379596056476411368831262392320000000").unwrap();
            assert_eq!(m.bit_length(), 896);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_1024(b: &mut Bencher) {
            let m: BigInt = str::parse("109175379332370811167728628129390764305676406997805846216644263719205294380444704345125974951708498876375436941932493630915136346984935259992421259591222990357198403617739536739192495783201936500406255369971622268782990346555658326035783303492057824725715044693252727793196332572871197862790983948697600000000").unwrap();
            assert_eq!(m.bit_length(), 1024);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_1280(b: &mut Bencher) {
            let m: BigInt = str::parse("11159786521638949063111676530081330557095127304043827586768868885330713040355866812502444336285627277554102293548054242071778656452262787088631467230450094195840972173312831618532761792907497971849407470539143466242472581931436192477989762316176801662953561092704922776628011688627324342023912470830789148798395797541846757863269596437206573449999381279287481256501238290186240000000000").unwrap();
            assert_eq!(m.bit_length(), 1280);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_1536(b: &mut Bencher) {
            let m: BigInt = str::parse("1140741034930644304091990219498801558933717687672771499989470642370364884836317524093419287538661536185379880193915444894231055351658947471828864523019531064788649954130080585389087366893220480286111129386725597864698747226863703441862381673649172685025052588791890974856933850860066378364603705951703997372212290133901343779346246624686343953553027072284643729154727263388882964780165279471557555000607099740718868227790160995745896395024531797835776000000000000").unwrap();
            assert_eq!(m.bit_length(), 1535);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_1792(b: &mut Bencher) {
            let m: BigInt = str::parse("116605286870982844093620522134643892284528848335640127585202150649286515178342135291642658524984574926089536696963778911744093971630059697111362328592828770568517573560215859046778451630201271855171531871485195657528834933053222412908437649760953799980041502898034150617058056644764324645770985102143998569348759807173056355236570472604881507371125755780841882047257906921419881198945554528297524801350524050966579125958178776219591412559516230979504302765155577406139840672473151563315342062386643270511515244952405569594982400000000000000").unwrap();
            assert_eq!(m.bit_length(), 1791);
            bench_core(b, &m);
        }

        #[bench]
        fn bench_2048(b: &mut Bencher) {
            let m: BigInt = str::parse("11919263452367059666018767206646516811042722401229904215556238457605297409757143070037879145312700685756611459168847326575559580253511486378534783141802930544523303603592033174844605626269857236296584909916935140188850184921847374612433948350120413737857810296652014840419928040409382335515954608864253937524757347592397440612028754280737096770230368277054931704290224165299843125505132905435715700230492710013707895760947527291072845887774637868810622595696796657661930103622876635161471752312090759515854195748854119478950486548009570476057421635277994889983439015042209811043389216159647508736245760000000000000000").unwrap();
            assert_eq!(m.bit_length(), 2047);
            bench_core(b, &m);
        }

    }

}
