/*
private static int hashCode(int result, byte[] a, int fromIndex, int length) {
        int end = fromIndex + length;
        for (int i = fromIndex; i < end; i++) {
            result = 31 * result + a[i];
        }
        return result;
    }

     */
pub fn java_str_hash_code(s: &str) -> i32 {
    let mut h: i32 = 0;
    for c in s.chars() {
        h = h.wrapping_mul(31).wrapping_add(c as i32);
    }
    h
}

/*
    public static RandomSeed.XoroshiroSeed createXoroshiroSeed(String seed) {
        byte[] bs = MD5_HASH.hashString(seed, Charsets.UTF_8).asBytes();
        long l = Longs.fromBytes(bs[0], bs[1], bs[2], bs[3], bs[4], bs[5], bs[6], bs[7]);
        long m = Longs.fromBytes(bs[8], bs[9], bs[10], bs[11], bs[12], bs[13], bs[14], bs[15]);
        return new RandomSeed.XoroshiroSeed(l, m);
    }
*/

pub fn xoroshiro_seed(seed: &str) -> (u64, u64) {
    let hash = md5::compute(seed);
    let l = u64::from_be_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ]);
    let m = u64::from_be_bytes([
        hash[8], hash[9], hash[10], hash[11], hash[12], hash[13], hash[14], hash[15],
    ]);
    (l, m)
}
