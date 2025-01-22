# PQ-FastScan
PQ-FastScan is a Rust implementation for fast vector quantization and search using Product Quantization (PQ). It supports 8-bit PQ using AVX512 SIMD instructions, which significantly accelerates the computation by leveraging advanced vector extensions.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Rust (latest stable version)
- Cargo (comes with Rust)
- HDF5 library

### Installation Steps

1. Install Rust and Cargo:
    ```sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

2. Install HDF5 library:
(On Linux)
    ```sh
    sudo apt-get update
    sudo apt-get install -y libhdf5-dev
    ```

3. Install HDF5 library (On MacOS):
    ```sh
    brew install hdf5
    ```

4. Clone the repository:
    ```sh
    git clone https://github.com/arkrishn94/pq-fastscan.git
    cd pq-fastscan
    ```

5. Build the project:
    ```sh
    cargo build --release
    ```

## Running the Code

1. Ensure you have the test data in `.hdf5` format with the following keys:
    - `train`: Data to be indexed and used for training
    - `test`: Queries to be searched.
    Make sure the data is in float32 format.

2. Run the code:
    ```sh
    cargo run --release -- <file_name> <num_PQ_sections> <top_k>
    ```

Replace `<file_name>` with the path to your data file, `<num_PQ_sections>` with the number of PQ sections, and `<top_k>` with the number of top results to retrieve.


This will execute the code in [main.rs] and perform the vector quantization and search operations.

### Example Commands and Files

To run PQ-FastScan, you can use an example HDF5 file from the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) repository. Here is an example command:

1. Download an example HDF5 file:
    ```sh
    wget https://github.com/erikbern/ann-benchmarks/raw/master/data/glove-100-angular.hdf5 -P data/
    ```

2. Run PQ-FastScan with the downloaded file:
    ```sh
    cargo run --release -- data/glove-100-angular.hdf5 25 10
    ```

In this example:
- `data/glove-100-angular.hdf5` is the path to the HDF5 file.
- `25` is the number of PQ sections.
- `10` is the number of top results to retrieve.

For more example datasets, you can visit the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) repository.