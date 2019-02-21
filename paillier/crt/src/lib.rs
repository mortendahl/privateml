#![feature(test)]

extern crate gmp;
extern crate rayon;
extern crate test;

use gmp::mpz::Mpz as BigInt;
use test::Bencher;

static P: &'static str = "148677972634832330983979593310074301486537017973460461278300587514468301043894574906886127642530475786889672304776052879927627556769456140664043088700743909632312483413393134504352834240399191134336344285483935856491230340093391784574980688823380828143810804684752914935441384845195613674104960646037368551517";
static Q: &'static str = "158741574437007245654463598139927898730476924736461654463975966787719309357536545869203069369466212089132653564188443272208127277664424448947476335413293018778018615899291704693105620242763173357203898195318179150836424196645745308205164116144020613415407736216097185962171301808761138424668335445923774195463";
static X: &'static str = "8116954461269652085230775933492366253929619979964246027246617328236243795946267984122836662596238827711003162168747438362516197517513090468979133736169125128476037082825330864610731186580002727070392849209478375921588198871833235568390694413294064765159446378195902634553122666031519508653458373801364626796907156918652837961453184912515251722492496853769056007056222605281803303377223245158930080581216344814858180859850388248516110876304421350734473865383370328091654104048265291335553686536171725033973437997155180998731315175192344098133206744942235940959001435284014629247610235987290001278275024859859011530610";
static R: &'static str = "23469954723780858594246099985284351679616402035799877801904657600748845577732527349622767465073206402501465434997097886062288454586744002070231423281154066427914375124691393663408153322045175238293238130584029098551997891083995941940072522297040482563102994301991914687415608932934232322244422743863612925485379738169539911563673020855346364903207395531099432657381392680821739845416005854261986810418869105342379811199853817109183567590778515098199155952701550697969915592026323126896564744857660315540347612223681727710811401589383421009333736638539423942798822879086915723828754886239159868603235567977031107975333";
static C: &'static str = "257203447105391916804089534214900683312963097096490235488958404866382741464385583184601128552670618981990000105037083690090594298564475202598013462067431119022723496775733949006049841771523035314712582893707414778394151736782745046995179058013948782711907055567514176057362937274003340106139237726460592991291649036854647465105678112034545238612525869223794770077034914371840530192251732830868984008901241006294086749915286415154943481920642073417739888367146198161135533211217852125267939460775999200943311678088646957279291231065720239752089787957176968565441059132771040186019584522752156210356879184733001364159458400826829676108356363700260706055423980900810086835364443358559049002895766426201073894808737170968355348275995156366869463099375320511865686019912874183927654191322970227080404808169296681251742918581398335645810766224246925708682914546637276241236065831895597376590899342810804823166192194221790279822050700148335320259676027171464292155454236417540711787990414904918789189297629562318139402290148344330957118511748943890681355239458737273975338090521441500322894883091962991413705362193011723285869089808996484682861997586072545454669866370589115975111183983919754128226878169359264094995923843080089523149217085";

fn modinv(a: &BigInt, modulus: &BigInt) -> BigInt {
    a.invert(modulus).unwrap()
}

fn modpow(base: &BigInt, exponent: &BigInt, modulus: &BigInt) -> BigInt {
    base.powm(exponent, modulus)
}

fn l(u: &BigInt, n: &BigInt) -> BigInt {
    (u - 1) / n
}

fn h(p: &BigInt, pp: &BigInt, n: &BigInt) -> BigInt {
    // compute g^{p-1} mod p^2
    let gp = (1 - n) % pp;
    // compute L_p(.)
    let lp = l(&gp, p);
    // compute L_p(.)^{-1}
    modinv(&lp, p)
}

pub struct EncryptionKey {
    n: BigInt, nn: BigInt,
    p: BigInt, pp: BigInt,
    q: BigInt, qq: BigInt,
}

impl EncryptionKey {
    fn from(p: BigInt, q: BigInt) -> EncryptionKey {
        let n = &p * &q;
        let nn = &n * &n;
        let pp = &p * &p;
        let qq = &q * &q;
        EncryptionKey { 
            n, nn,
            p, pp,
            q, qq
        }
    }
}

pub struct DecryptionKey {
    n: BigInt, nn: BigInt,
    p: BigInt, pp: BigInt,
    q: BigInt, qq: BigInt,
    g: BigInt,
    n_order: BigInt, n_order_inv: BigInt,
    p_order: BigInt, p_order_inv: BigInt,
    q_order: BigInt, q_order_inv: BigInt,
}

impl DecryptionKey {
    fn from(p: BigInt, q: BigInt) -> DecryptionKey {
        let n = &p * &q;
        let nn = &n * &n;
        let pp = &p * &p;
        let qq = &q * &q;

        let g = 1 + &n;

        let n_order = (&p - 1) * (&q - 1);
        let n_order_inv = modinv(&n_order, &n);

        let p_order = &p - 1;
        let p_order_inv = modinv(&p_order, &p);

        let q_order = &q - 1;
        let q_order_inv = modinv(&q_order, &q);

        DecryptionKey { 
            n, nn,
            p, pp,
            q, qq,
            g,
            n_order, n_order_inv,
            p_order, p_order_inv,
            q_order, q_order_inv
        }
    }
}

mod plain {

    use super::*;

    pub fn encrypt(x: &BigInt, r: &BigInt, ek: &EncryptionKey) -> BigInt {
        let rm = modpow(r, &ek.n, &ek.nn);
        let gx = (1 + x * &ek.n) % &ek.nn;
        (gx * rm) % &ek.nn
    }

    #[test]
    fn test_encrypt() {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let r: BigInt = str::parse(R).unwrap();
        let c: BigInt = str::parse(C).unwrap();
        let ek = EncryptionKey::from(p, q);

        assert_eq!(encrypt(&x, &r, &ek), c);
    }

    #[bench]
    fn bench_encrypt(b: &mut Bencher) {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let r: BigInt = str::parse(R).unwrap();
        let ek = EncryptionKey::from(p, q);

        b.iter(|| {
            let _ = encrypt(&x, &r, &ek);
        });
    }

    pub fn decrypt(c: &BigInt, dk: &DecryptionKey) -> BigInt {
        let d = &dk.n_order * &dk.n_order_inv;
        let gx = modpow(c, &d, &dk.nn);
        (gx - 1) / &dk.n
    }

    #[test]
    fn test_decrypt() {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let c: BigInt = str::parse(C).unwrap();
        let dk = DecryptionKey::from(p, q);
        
        assert_eq!(decrypt(&c, &dk), x);
    }

    #[bench]
    fn bench_decrypt(b: &mut Bencher) {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let c: BigInt = str::parse(C).unwrap();
        let dk = DecryptionKey::from(p, q);

        b.iter(|| {
            let _ = decrypt(&c, &dk);
        });
    }

}

mod crt {

    use super::*;

    fn crt(x1: &BigInt, x2: &BigInt, m1: &BigInt, m2: &BigInt) -> BigInt {
        let mut diff = (x2 - x1) % m2;
        if diff < BigInt::zero() {
            diff += m2;
        }
        let u = (diff * modinv(m1, m2)) % m2;
        x1 + (u * m1)
    }

    #[test]
    fn test_crt() {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let r: BigInt = str::parse(R).unwrap();
        let c: BigInt = str::parse(C).unwrap();

        let pp = &p * &p;
        let qq = &q * &q;

        let (x1, x2) = (&x % &p, &x % &q);
        assert_eq!(crt(&x1, &x2, &p, &q), x);

        let (r1, r2) = (&r % &p, &r % &q);
        assert_eq!(crt(&r1, &r2, &p, &q), r);

        let (c1, c2) = (&c % &pp, &c % &qq);
        assert_eq!(crt(&c1, &c2, &pp, &qq), c);
    }

    fn encrypt_component(x: &BigInt, r: &BigInt, ek: &EncryptionKey, mm: &BigInt) -> BigInt {
        let rm = modpow(r, &ek.n, mm);
        let gx = (1 + x * &ek.n) % mm;
        (gx * rm) % mm
    }

    pub fn encrypt(
        x: &BigInt,
        r: &BigInt,
        ek: &EncryptionKey,
    ) -> BigInt {
        let (xp, xq) = (x % &ek.p, x % &ek.q);
        let (rp, rq) = (r % &ek.p, r % &ek.q);
        let cp = encrypt_component(&xp, &rp, &ek, &ek.pp);
        let cq = encrypt_component(&xq, &rq, &ek, &ek.qq);
        crt(&cp, &cq, &ek.pp, &ek.qq)
    }

    #[test]
    fn test_encrypt() {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let r: BigInt = str::parse(R).unwrap();
        let c: BigInt = str::parse(C).unwrap();
        let ek = EncryptionKey::from(p, q);

        assert_eq!(encrypt(&x, &r, &ek), c);
    }

    #[bench]
    fn bench_encrypt(b: &mut Bencher) {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let r: BigInt = str::parse(R).unwrap();
        let ek = EncryptionKey::from(p, q);

        b.iter(|| {
            let _ = encrypt(&x, &r, &ek);
        });
    }

    fn decrypt_component(
        c: &BigInt,
        dk: &DecryptionKey,
        m: &BigInt,
        mm: &BigInt,
        m_order: &BigInt,
        m_order_inv: &BigInt,
    ) -> BigInt {
        let dm = modpow(c, m_order, mm);
        let lm = l(&dm, m);
        let bar = modpow(&dk.g, m_order, mm);
        let foo = l(&bar, m);
        let hm = modinv(&foo, m);
        (lm * hm) % m
    }

    pub fn decrypt(c: &BigInt, dk: &DecryptionKey) -> BigInt {
        let (cp, cq) = (c % &dk.pp, c % &dk.qq);
        let mp = decrypt_component(&cp, dk, &dk.p, &dk.pp, &dk.p_order, &dk.p_order_inv);
        let mq = decrypt_component(&cq, dk, &dk.q, &dk.qq, &dk.q_order, &dk.q_order_inv);
        crt(&mp, &mq, &dk.p, &dk.q)
    }

    #[test]
    fn test_decrypt() {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let x: BigInt = str::parse(X).unwrap();
        let c: BigInt = str::parse(C).unwrap();
        let dk = DecryptionKey::from(p, q);

        assert_eq!(decrypt(&c, &dk), x);
    }

    #[bench]
    fn bench_decrypt(b: &mut Bencher) {
        let p: BigInt = str::parse(P).unwrap();
        let q: BigInt = str::parse(Q).unwrap();
        let c: BigInt = str::parse(C).unwrap();
        let dk = DecryptionKey::from(p, q);

        b.iter(|| {
            let _ = decrypt(&c, &dk);
        });
    }

    mod parallel {

        use super::Bencher;
        use super::{BigInt, P, Q, X, R, C};
        use super::{crt};
        use super::{EncryptionKey, encrypt_component};
        use super::{DecryptionKey, decrypt_component};

        use rayon::join;

        pub fn encrypt(x: &BigInt, r: &BigInt, ek: &EncryptionKey) -> BigInt {
            let (cp, cq) = join(
                || {
                    let xp = x % &ek.pp;
                    let rp = r % &ek.pp;
                    encrypt_component(&xp, &rp, ek, &ek.pp)
                },
                || {
                    let xq = x % &ek.qq;
                    let rq = r % &ek.qq;
                    encrypt_component(&xq, &rq, ek, &ek.qq)
                },
            );
            crt(&cp, &cq, &ek.pp, &ek.qq)
        }

        #[bench]
        fn bench_encrypt(b: &mut Bencher) {
            let p: BigInt = str::parse(P).unwrap();
            let q: BigInt = str::parse(Q).unwrap();
            let x: BigInt = str::parse(X).unwrap();
            let r: BigInt = str::parse(R).unwrap();
            let ek = EncryptionKey::from(p, q);

            b.iter(|| {
                let _ = encrypt(&x, &r, &ek);
            });
        }

        pub fn decrypt(c: &BigInt, dk: &DecryptionKey) -> BigInt {
            let (mp, mq) = join(
                || {
                    let cp = c % &dk.pp;
                    decrypt_component(&cp, dk, &dk.p, &dk.pp, &dk.p_order, &dk.p_order_inv)
                },
                || {
                    let cq = c % &dk.qq;
                    decrypt_component(&cq, dk, &dk.q, &dk.qq, &dk.q_order, &dk.q_order_inv)
                },
            );
            crt(&mp, &mq, &dk.p, &dk.q)
        }

        #[bench]
        fn bench_decrypt(b: &mut Bencher) {
            let p: BigInt = str::parse(P).unwrap();
            let q: BigInt = str::parse(Q).unwrap();
            let c: BigInt = str::parse(C).unwrap();
            let dk = DecryptionKey::from(p, q);

            b.iter(|| {
                let _ = decrypt(&c, &dk);
            });
        }
    }
}
