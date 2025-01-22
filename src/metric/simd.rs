// AVX-512 imports only when available
#[cfg(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
use core::arch::x86_64::*;

// Type definitions that work across all platforms
#[cfg(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
pub type Simd64uint8 = __m512i;

#[cfg(not(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
#[derive(Clone, Copy, Debug)]
pub struct Simd64uint8(pub [u8; 64]);

#[cfg(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
pub type Simd64bitMask = __mmask64;

#[cfg(not(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
pub type Simd64bitMask = u64;

// AVX-512 Implementations
#[cfg(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
impl Simd64uint8 {
    #[inline]
    pub fn to_array(self) -> [u8; 64] {
        let mut result = [0u8; 64];
        unsafe {
            _mm512_storeu_epi8(result.as_mut_ptr() as *mut i8, self);
        }
        result
    }
}

// AVX-512 Functions
#[cfg(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
mod avx512_impl {
    use super::*;

    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub unsafe fn simd_load_64uint8(ptr: &[u8]) -> Simd64uint8 {
        _mm512_loadu_epi8(ptr.as_ptr() as *const i8)
    }

    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub unsafe fn simd_load_zero_64uint8() -> Simd64uint8 {
        _mm512_setzero_si512()
    }

    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub unsafe fn simd_getblk64mask(codes: Simd64uint8) -> Simd64bitMask {
        _mm512_movepi8_mask(codes)
    }

    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
    pub unsafe fn simd_masked_lookup_64uint8(
        codes: Simd64uint8,
        qlut0: Simd64uint8,
        qlut1: Simd64uint8,
    ) -> Simd64uint8 {
        _mm512_permutex2var_epi8(qlut0, codes, qlut1)
    }

    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub unsafe fn simd_blend_64uint8(
        mask: Simd64bitMask,
        qlut0: Simd64uint8,
        qlut1: Simd64uint8,
    ) -> Simd64uint8 {
        _mm512_mask_blend_epi8(mask, qlut0, qlut1)
    }

    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub unsafe fn simd_add_64uint8(partial0: Simd64uint8, partial1: Simd64uint8) -> Simd64uint8 {
        _mm512_adds_epu8(partial0, partial1)
    }

    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub unsafe fn simd_set_arr_64uint8(result: *mut u8, acc: Simd64uint8) {
        _mm512_storeu_epi8(result as *mut i8, acc)
    }
}

// Fallback Implementations when AVX512 is not available.
#[cfg(not(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
impl Simd64uint8 {
    #[inline]
    pub fn to_array(self) -> [u8; 64] {
        self.0
    }
}

#[cfg(not(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
mod fallback_impl {
    use super::*;

    #[inline]
    pub fn simd_load_64uint8(ptr: &[u8]) -> Simd64uint8 {
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&ptr[..64]);
        Simd64uint8(arr)
    }

    #[inline]
    pub fn simd_load_zero_64uint8() -> Simd64uint8 {
        Simd64uint8([0u8; 64])
    }

    #[inline]
    pub fn simd_getblk64mask(codes: Simd64uint8) -> Simd64bitMask {
        // Convert the highest bit of each byte into a bit mask
        let mut mask: u64 = 0;
        for (i, &byte) in codes.0.iter().enumerate() {
            if (byte & 0x80) != 0 {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[inline]
    pub fn simd_masked_lookup_64uint8(
        codes: Simd64uint8,
        qlut0: Simd64uint8,
        qlut1: Simd64uint8,
    ) -> Simd64uint8 {
        let mut result = [0u8; 64];
        for i in 0..64 {
            let idx = codes.0[i] as usize;
            result[i] = if idx < 64 {
                qlut0.0[idx]
            } else {
                qlut1.0[idx - 64]
            };
        }
        Simd64uint8(result)
    }

    #[inline]
    pub fn simd_blend_64uint8(
        mask: Simd64bitMask,
        qlut0: Simd64uint8,
        qlut1: Simd64uint8,
    ) -> Simd64uint8 {
        let mut result = [0u8; 64];
        for i in 0..64 {
            result[i] = if ((mask >> i) & 1) != 0 {
                qlut1.0[i]
            } else {
                qlut0.0[i]
            };
        }
        Simd64uint8(result)
    }

    #[inline]
    pub fn simd_add_64uint8(partial0: Simd64uint8, partial1: Simd64uint8) -> Simd64uint8 {
        let mut result = [0u8; 64];
        for i in 0..64 {
            // Saturating addition
            result[i] = partial0.0[i].saturating_add(partial1.0[i]);
        }
        Simd64uint8(result)
    }

    #[inline]
    pub fn simd_set_arr_64uint8(result: *mut u8, acc: Simd64uint8) {
        unsafe {
            std::ptr::copy_nonoverlapping(acc.0.as_ptr(), result, 64);
        }
    }
}

// Export the appropriate implementation based on features
#[cfg(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
pub use avx512_impl::*;

#[cfg(not(all(
    feature = "nightly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
pub use fallback_impl::*;

// Runtime feature detection
#[cfg(target_arch = "x86_64")]
pub fn has_avx512_support() -> bool {
    if cfg!(all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512vbmi"))
    {
        return true;
    }
    if cfg!(feature = "nightly") {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
    } else {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512_support() -> bool {
    false
}