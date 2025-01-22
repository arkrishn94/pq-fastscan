// src/pq.rs

use crate::metric::simd::*;
use num::pow;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use crate::metric::utils::smallest_heap_with_existing;
use crate::metric::Metric;

const CORESET_SIZE: usize = 512;
const BLK_SIZE: usize = 64;
const FS8_KBITS: usize = 8; // 8 bits per section = 256 centers per section


// Search result structure
#[derive(Copy, Clone, PartialEq)]
pub struct SearchResult {
    pub idx: usize,
    pub distance: f32,
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// Centroids structure
#[derive(Clone)]
pub struct Centroids<M: Metric> {
    centers: Vec<Vec<f32>>,
    dimension: usize,
    _metric: PhantomData<M>,
}

impl<M: Metric> Centroids<M> {
    pub fn new(n_centers: usize, dimension: usize) -> Self {
        Self {
            centers: Vec::with_capacity(n_centers),
            dimension,
            _metric: PhantomData,
        }
    }

    pub fn insert(&mut self, center: &[f32]) -> Result<(), &'static str> {
        if center.len() != self.dimension {
            return Err("Invalid center dimension");
        }
        self.centers.push(center.to_vec());
        Ok(())
    }

    pub fn compute_all_distances(&self, query: &[f32], distances: &mut [f32], _weights: &[f32]) {
        for (i, center) in self.centers.iter().enumerate() {
            distances[i] = M::compute(query, center);
        }
    }
}

impl<M: Metric> std::ops::Index<usize> for Centroids<M> {
    type Output = Vec<f32>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.centers[index]
    }
}

// Per-query state for search
#[derive(Clone, Debug)]
pub struct PQFS8PerQueryState(pub Vec<f32>, pub Vec<u8>, pub f32, pub f32, pub Vec<Simd64uint8>);

// Main PQ FastScan implementation
pub struct PQFastScan8Quantizer<M: Metric> {
    pub n_sections: usize,
    pub centroids: Vec<Centroids<M>>,
    pub coreset_codes: Vec<u8>,
    pub is_trained: bool,
    pub dimension: usize,
    _pq_type: PhantomData<M>,
}

impl<M: Metric> PQFastScan8Quantizer<M> {
    pub fn new(dimension: usize, n_sections: usize) -> Self {
        assert!(n_sections > 0);
        Self {
            n_sections,
            centroids: vec![Centroids::<M>::new(pow(2_usize, FS8_KBITS), dimension / n_sections); n_sections],
            coreset_codes: vec![],
            is_trained: false,
            dimension,
            _pq_type: PhantomData,
        }
    }

    pub fn train_from_collection(&mut self, train_data: &[f32]) {
        assert_eq!(train_data.len() % self.dimension, 0);
        let train_size = train_data.len() / self.dimension;
        let k = pow(2_usize, FS8_KBITS);
        let sec_dim = self.dimension / self.n_sections;

        // Train each section
        for m in 0..self.n_sections {
            // Simplified k-means implementation for each section
            // In practice, you'd want to implement a proper k-means here
            let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
            
            // Initialize centroids with random vectors
            for _ in 0..k {
                let vec: Vec<f32> = (0..sec_dim).map(|_| rand::random()).collect();
                centroids.push(vec);
            }
            
            // Add centroids to the index
            for centroid in centroids {
                self.centroids[m].insert(&centroid).unwrap();
            }
        }

        // Create coreset codes
        let mut coreset_codes = Vec::with_capacity(CORESET_SIZE * self.n_sections);
        train_data
            .chunks_exact(self.dimension)
            .take(CORESET_SIZE)
            .for_each(|v| coreset_codes.extend_from_slice(&self.encode(v).0));

        self.coreset_codes = coreset_codes;
        self.is_trained = true;
    }

    pub fn encode(&self, vec: &[f32]) -> (Vec<u8>, Option<Vec<f32>>) {
        let sec_dim = self.dimension / self.n_sections;
        
        // Find closest centroid for each section
        let codes: Vec<u8> = self.centroids
            .iter()
            .enumerate()
            .map(|(m, idx)| {
                let start = m * sec_dim;
                let end = start + sec_dim;
                let mut min_dist = f32::INFINITY;
                let mut min_idx = 0;
                
                for (i, center) in idx.centers.iter().enumerate() {
                    let dist = M::compute(&vec[start..end], center);
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = i;
                    }
                }
                min_idx as u8
            })
            .collect();

        // Construct decoded vector
        let mut decoded = Vec::with_capacity(self.dimension);
        for (m, &code) in codes.iter().enumerate() {
            decoded.extend_from_slice(&self.centroids[m][code as usize]);
        }

        (codes, Some(decoded))
    }


    fn compute_lookup_table(&self, query: &[f32]) -> Vec<f32> {
        let k = pow(2_usize, FS8_KBITS);
        let chunk_size = self.dimension / self.n_sections;
        let mut dist_lut = vec![0f32; k * self.n_sections];

        for m in 0..self.n_sections {
            let q_sub = &query[m * chunk_size..(m + 1) * chunk_size];
            let start = m * k;
            self.centroids[m].compute_all_distances(q_sub, &mut dist_lut[start..start + k], &[1.0]);
        }

        dist_lut
    }



    /// Quantizes the lookup table of distances.
    ///
    /// This function takes a lookup table of distances and quantizes it into a vector of bytes.
    /// It also computes the scaling factor and shift value used for quantization.
    ///
    /// # Arguments
    ///
    /// * `dist_lut` - A slice of distances to be quantized.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `Vec<u8>` - The quantized lookup table.
    /// * `f32` - The scaling factor used for quantization.
    /// * `f32` - The shift value used for quantization.
    ///
    /// # Details
    ///
    /// The function first computes the minimum distance for each section of the lookup table.
    /// It then calculates the shift value as the sum of these minimum distances.
    /// The scaling factor is computed as `255 / (max_dist - shift)`, where `max_dist` is the maximum
    /// distance from the precomputed coreset codes.
    /// Finally, the function quantizes each distance in the lookup table using the formula:
    /// `quantized_distance = ceil((distance - min_distance) * scaling + 0.5)`.
    /// 
    /// In particular,
    /// Let d_j[k] be the distance to the k-th codeword in the j-th section (i.e. codebook)
    // and let d_max be some upper-bound of the distance from the NN to the query.
    //Then, first the following two quantities are computed,
    //      dist_shift = sum_{j = [1, m]} min_{k' in [0, 255]} d_j[k']
    //      dist_scaling = 255 / (d_max - dist_shift).
    //The byte-quantized distance d'_j[k] is then computed as follows,
    //      d'_j[k] = ceil (dist_scaling * (d_j[k] - min_{k' in [0, 255]} d_j[k']))
    // sum_j d'_j[k] = dist_scaling * (sum_j d_j[k] - dist_shift))
    fn quantize_lookup_table(&self, dist_lut: &[f32]) -> (Vec<u8>, f32, f32) {
        let k256 = pow(2_usize, FS8_KBITS);
        
        // Find min values for each section
        let mut mins = Vec::new();
        for chunk in dist_lut.chunks(k256) {
            mins.push(chunk.iter().copied().fold(f32::INFINITY, f32::min));
        }

        let shift = mins.iter().sum();
        
        // Compute scaling factor
        let max_dist = self.compute_max_pq_distance(dist_lut, &self.coreset_codes);
        let scaling = 255.0 / (max_dist - shift);

        // Quantize the lookup table
        let quantized = dist_lut
            .chunks(k256)
            .zip(mins.iter())
            .flat_map(|(chunk, &min_val)| {
                chunk
                    .iter()
                    .map(|&d| ((d - min_val) * scaling + 0.5) as u8)
                    .collect::<Vec<_>>()
            })
            .collect();

        (quantized, scaling, shift)
    }

    fn compute_max_pq_distance(&self, dist_lut: &[f32], codes: &[u8]) -> f32 {
        let mut max_dist = f32::NEG_INFINITY;

        for chunk in codes.chunks(self.n_sections) {
            let dist = chunk
                .iter()
                .enumerate()
                .map(|(j, &c)| dist_lut[j * 256 + c as usize])
                .sum();
            max_dist = max_dist.max(dist);
        }

        max_dist
    }
}


///Search functions for PQFastScan8Quantizer
impl<M: Metric> PQFastScan8Quantizer<M> {
    pub fn search(&self, query: &[f32], ids: &[u32], codes: &[u8], k: usize) -> BinaryHeap<SearchResult> {
        let state = self.prepare_search(query);
        let mut heap = BinaryHeap::new();
        
        self.heap_get_candidates(codes, &ids, query, &state, k, &mut heap);
        heap
    }

    pub fn prepare_search(&self, query: &[f32]) -> PQFS8PerQueryState {
        let dist_lut = self.compute_lookup_table(query);
        let (quantized_lut, dist_scaling, dist_shift) = self.quantize_lookup_table(&dist_lut);
        
        let quantized_lut_simd = 
            quantized_lut
                .chunks_exact(BLK_SIZE)
                .map(|chunk| { simd_load_64uint8(chunk) })
                .collect();

        PQFS8PerQueryState(
            dist_lut,
            quantized_lut,
            dist_scaling,
            dist_shift,
            quantized_lut_simd,
        )
    }

    pub fn heap_get_candidates(
        &self,
        codes: &[u8],
        ids: &[u32],
        _q: &[f32],
        state: &PQFS8PerQueryState,
        top_k: usize,
        heap: &mut BinaryHeap<SearchResult>,
    ) {
        let k256 = pow(2_usize, FS8_KBITS);
        assert_eq!(k256, 256);

        let code_size: usize = self.n_sections;
        let n_points = ids.len();

        let (quantized_lut, lut_scaling, lut_shift) = (
            state.1.as_slice(),
            state.2,
            state.3,
        );
        let quantized_lut_simd = state.4.as_slice();
        assert_eq!(quantized_lut_simd.len(), code_size * 4);

        let mut dists_arr = vec![0_u8; BLK_SIZE];

        codes
            .chunks_exact(BLK_SIZE * self.n_sections)
            .zip(ids.chunks_exact(BLK_SIZE))
            .for_each(|(blk_codes, blk_ids)| {
                self.simd_blk64_acc_kernel1(
                    blk_codes,
                    quantized_lut_simd,
                    dists_arr.as_mut_ptr(),
                );

                let results = dists_arr.iter().zip(blk_ids).map(|(&distance, &idx)| {
                    SearchResult {
                        idx: idx as usize,
                        distance: (distance as f32 / lut_scaling) + lut_shift,
                    }
                });

                smallest_heap_with_existing(top_k, results, heap);
            });
    
        //For remaining points, not in a BLK_SIZE multiple, compute distances one by one.
        let dist_lut = state.0.as_slice();
        let remainder_points = n_points % BLK_SIZE;
        let results = (n_points - remainder_points..).map(|id| SearchResult {
            idx: unsafe { *ids.get_unchecked(id) } as usize,
            distance: unsafe {
                codes[id * code_size..(id + 1) * code_size]
                    .iter()
                    .enumerate()
                    .map(|(mid, m_code)| {
                        dist_lut.get_unchecked(mid * k256 + *m_code as usize)
                    })
                    .sum::<f32>()
            },
        });

        smallest_heap_with_existing(top_k, results, heap);
        
    }

    ///SIMD Kernel that computes the distance between the query and the BLK_SIZE points.
    /// The distances are computed section by section for BLK_SIZE points at once. 
    /// The distances are accumulated in u8s, as saturated adds.
    /// For each section j, let C = [c0j, c1j, ..., c64j] be the codes for the 64 points. 
    /// Let [q0j, q1j, ..., q255j] be the quantized distances for the 256 codewords in the j-th section. They are split into 4 arrays of 64 elements each : Q1, Q2, Q3, Q4.
    ///      First, we use the bottom 7-bits of cij to index into Q1 and Q2 to get the quantized distances from [Q1; Q2]
    ///      Next, we do the same to get the quantized distances from [Q3; Q4].
    ///      Finally, we use the top 1-bit of cij to select between the quantized distances from [Q1; Q2] and [Q3; Q4] (using a blend operation)
    ///     The 64 distances are then accumulated in u8s, as saturated adds.
    #[inline]
    fn simd_blk64_acc_kernel1(
        &self,
        blk_codes_t: &[u8],
        quantized_lut: &[Simd64uint8],
        result_ptr: *mut u8,
    ) {
        let code_size = self.n_sections;
        assert_eq!(blk_codes_t.len(), code_size * BLK_SIZE);

        // accumulations happening in u8s, as saturated adds
        let mut acc = unsafe { simd_load_zero_64uint8() };

        blk_codes_t
            .chunks_exact(BLK_SIZE)
            .enumerate()
            .for_each(|(sec, blk)| unsafe {
                let codes = simd_load_64uint8(blk);
                let code_mask = simd_getblk64mask(codes);

                let q_lut00 = &quantized_lut[4 * sec];
                let q_lut01 = &quantized_lut[4 * sec + 1];
                let q_lut02 = &quantized_lut[4 * sec + 2];
                let q_lut03 = &quantized_lut[4 * sec + 3];

                let partial01 = simd_masked_lookup_64uint8(codes, *q_lut00, *q_lut01);
                let partial23 = simd_masked_lookup_64uint8(codes, *q_lut02, *q_lut03);

                let blended = simd_blend_64uint8(code_mask, partial01, partial23);

                acc = simd_add_64uint8(acc, blended);
            });

        unsafe {
            simd_set_arr_64uint8(result_ptr, acc);
        }
    }
}