use hdf5::File;
use std::time::Instant;
use std::env;

pub mod metric;
pub mod pqfs;

use pqfs::PQFastScan8Quantizer;
use metric::EuclideanMetric;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <file_name> <num_PQ_sections> <top_k>", args[0]);
        std::process::exit(1);
    }
    let file_name = &args[1];

    // Open HDF5 file containing test dataset
    let file = File::open(file_name)?;
    
    // Read training, base and query vectors
    let base_dset = file.dataset("train")?;
    let query_dset = file.dataset("query")?;
    
    let base_data: Vec<f32> = base_dset.read_raw()?;
    let query_data: Vec<f32> = query_dset.read_raw()?;
    
    // Get dimensions
    let n_base = base_dset.shape()[0];
    let n_query = query_dset.shape()[0];
    let dim = base_dset.shape()[1];
    
    println!("Dataset dimensions:");
    println!("Base: {} x {}", n_base, dim);
    println!("Query: {} x {}", n_query, dim);
    
    // Initialize PQ quantizer
    let n_sections = args[2].parse().unwrap(); // Number of PQ sections
    let mut quantizer = PQFastScan8Quantizer::<EuclideanMetric>::new(dim, n_sections);
    
    // Train the quantizer
    println!("Training PQ quantizer...");
    let start = Instant::now();
    quantizer.train_from_collection(&base_data);
    println!("Training took: {:?}", start.elapsed());
    
    // Encode base vectors
    println!("Encoding base vectors...");
    let start = Instant::now();
    let mut codes = Vec::with_capacity(n_base * n_sections);
    for vec in base_data.chunks_exact(dim) {
        let (code, _) = quantizer.encode(vec);
        codes.extend_from_slice(&code);
    }
    println!("Encoding took: {:?}", start.elapsed());
    
    // Search parameters
    let k = args[3].parse().unwrap(); // Number of nearest neighbors to find

    let ids: Vec<u32> = (0..n_base as u32).collect();
    
    // Search queries
    println!("Searching {} nearest neighbors for {} queries...", k, n_query);
    let start = Instant::now();
    
    let mut total_query_time = 0.0;
    for (_, query) in query_data.chunks_exact(dim).enumerate() {
        let query_start = Instant::now();

        // Search using PQ
        
        quantizer.search(query, &ids, &codes, k);
        
        let query_time = query_start.elapsed();
        total_query_time += query_time.as_secs_f64();
    }
    
    let total_time = start.elapsed();
    println!("Average query time: {:.3} ms", (total_query_time * 1000.0) / n_query as f64);
    println!("Total search time: {:?}", total_time);
    
    Ok(())
}