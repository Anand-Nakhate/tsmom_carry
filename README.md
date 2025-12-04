# TSMOM Carry Strategy Data Pipeline


## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/tsmom_carry.git
    cd tsmom_carry
    ```

2.  **Install dependencies**:
    ```bash
    make install
    ```

3.  **Set API Key**:
    ```bash
    export DATABENTO_API_KEY=your_api_key
    ```

## Usage

### Download Data
Download the defined universe of futures contracts.
```bash
make download
```

### Process Data
Process raw files into a master dataset (`master_dataset.csv`).
```bash
make process
```

### Run Tests
Execute the test suite to verify integrity.
```bash
make test
```

## License

Proprietary and Confidential.
