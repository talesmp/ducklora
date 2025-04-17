from pathlib import Path
import random
import duckdb
import json
import time

def contains_chinese(text: str) -> bool:
    """
    Return True if the text contains any Chinese characters.
    Unicode range for common Chinese characters is from U+4E00 to U+9FFF.
    """
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def split_jsonl_file(input_path: Path, output_dir: Path, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, sample_percentage: int = 100, seed: int = None):
    """
    Splits a JSON Lines file into 3 files: train.jsonl, valid.jsonl, and test.jsonl.
    The split uses the specified proportions (default 70%/15%/15%). 
    Additionally, only a given percentage (sample_percentage, from 1 to 100) of the original lines will be processed and written out.
    
    For example, if the original file is 5.0GB and sample_percentage is set to 60, only about 60% of the lines will be processed (resulting in roughly 3.0GB total output, split between the 3 files).

    Parameters:
        input_path (Path): The path to the input JSON Lines file.
        output_dir (Path): The directory where the split files will be saved.
        train_ratio (float): Proportion of lines to assign to training (default 0.7).
        valid_ratio (float): Proportion of lines to assign to validation (default 0.15).
        test_ratio (float): Proportion of lines to assign to testing (default 0.15).
        sample_percentage (int): How much of the original file to use, from 1 to 100 (default 100).
        seed (int, optional): Random seed for reproducibility.
    """
    # Normalize ratios in case they don't add up exactly to 1.0
    total = train_ratio + valid_ratio + test_ratio
    train_ratio /= total
    valid_ratio /= total
    test_ratio /= total

    # Compute the sampling probability
    sample_prob = sample_percentage / 100.0

    if seed is not None:
        random.seed(seed)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    test_path = output_dir / "test.jsonl"

    # Open input and output files (streaming mode)
    with input_path.open("r", encoding="utf-8") as infile, \
         train_path.open("w", encoding="utf-8") as ftrain, \
         valid_path.open("w", encoding="utf-8") as fvalid, \
         test_path.open("w", encoding="utf-8") as ftest:
        for line in infile:
            # Load the JSON object and extract only the text field
            data = json.loads(line)
            text_value = data.get("text", "")

            # Skip this record if the text contains any Chinese characters.
            if contains_chinese(text_value):
                continue

            # Decide whether to include this line based on the sampling probability.
            if random.random() > sample_prob:
                continue

            new_data = {"text": text_value}
            json_line = json.dumps(new_data)
            
            # Determine which file to write to using the original ratios
            r = random.random()
            if r < train_ratio:
                ftrain.write(json_line + "\n")
            elif r < train_ratio + valid_ratio:
                fvalid.write(json_line + "\n")
            else:
                ftest.write(json_line + "\n")

def split_jsonl_file_duckdb(
    input_path: Path,
    output_dir: Path,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed: int = None,
    batch_size: int = 1000,
):
    """
    Splits a single JSON Lines file into three files (train.jsonl, valid.jsonl, test.jsonl)
    using a 70/15/15 proportion (by default). This implementation uses DuckDB to read the
    JSONL file efficiently and stream the data in batches.
    
    Parameters:
        input_path (Path): Path to the input JSONL file.
        output_dir (Path): Directory where the split files will be saved.
        train_ratio (float): Fraction of data for training (default 0.7).
        valid_ratio (float): Fraction of data for validation (default 0.15).
        test_ratio (float): Fraction of data for testing (default 0.15).
        seed (int, optional): Random seed for reproducibility.
        batch_size (int): Number of rows to fetch per batch.
    """
    # Warn if a seed is provided because DuckDB doesn't support seeding random()
    if seed is not None:
        print("Warning: DuckDB does not support setting a random seed; the provided seed will be ignored.")

    # If the input_path is not absolute, resolve it relative to the script's directory.
    if not input_path.is_absolute():
        input_path = (Path(__file__).parent / input_path).resolve()

    # Ensure the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    test_path = output_dir / "test.jsonl"

    # Connect to DuckDB (in-memory)
    con = duckdb.connect(database=":memory:")

    # Create a view directly on the JSONL file using DuckDB's JSON reader.
    # read_json_auto infers the schema and streams the file.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW data_view AS 
        SELECT * FROM read_ndjson('{str(input_path.resolve())}', maximum_object_size = 40000000)
        """
    )

    # Add a random number column to each row.
    con.execute(
        """
        CREATE OR REPLACE VIEW data_with_rand AS 
        SELECT *, random() AS rnd FROM data_view
        """
    )

    # Define SQL queries for each split.
    train_query = f"SELECT * FROM data_with_rand WHERE rnd < {train_ratio}"
    valid_query = f"SELECT * FROM data_with_rand WHERE rnd >= {train_ratio} AND rnd < {train_ratio + valid_ratio}"
    test_query  = f"SELECT * FROM data_with_rand WHERE rnd >= {train_ratio + valid_ratio}"

    def write_jsonl(query: str, output_file: Path):
        cur = con.cursor()
        cur.execute(query)
        # Get column names from the query result.
        columns = [desc[0] for desc in cur.description]
        with output_file.open("w", encoding="utf-8") as f:
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    # Build a dictionary for each row.
                    record = dict(zip(columns, row))
                    # Remove the 'rnd' column from output.
                    record.pop("rnd", None)
                    f.write(json.dumps(record) + "\n")

    # Write out each split.
    write_jsonl(train_query, train_path)
    write_jsonl(valid_query, valid_path)
    write_jsonl(test_query, test_path)

    con.close()

def split_jsonl_file_duckdb_improved(
    input_path: Path,
    output_dir: Path,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed: int = None,
):
    """
    Splits a JSON Lines file into three files using DuckDB’s native export (COPY).
    This improved version:
      1. Uses DuckDB’s COPY to write query results directly to disk,
         avoiding the Python loop and repeated JSON (de)serialization.
      2. Eliminates the batch fetch overhead.
      3. Removes the extra "rnd" column by dynamically retrieving the original
         column names from the JSON data.
    
    Parameters:
        input_path (Path): Path to the input JSONL file.
        output_dir (Path): Directory where the split files will be saved.
        train_ratio (float): Fraction of data for training.
        valid_ratio (float): Fraction of data for validation.
        test_ratio (float): Fraction of data for testing.
        seed (int, optional): Random seed (ignored in DuckDB).
    """
    if seed is not None:
        print("Warning: DuckDB does not support setting a random seed; the provided seed will be ignored.")

    # Resolve the input path if necessary.
    if not input_path.is_absolute():
        input_path = (Path(__file__).parent / input_path).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    test_path = output_dir / "test.jsonl"

    # Connect to DuckDB (in-memory)
    con = duckdb.connect(database=":memory:")

    # Create a view on the JSONL file.
    con.execute(f"""
        CREATE OR REPLACE VIEW data_view AS 
        SELECT * FROM read_ndjson('{str(input_path.resolve())}', maximum_object_size = 40000000)
    """)

    # Get the original column names (without the extra random column) from data_view.
    cols_info = con.execute("PRAGMA table_info('data_view')").fetchall()
    print(cols_info)
    columns = [row[1] for row in cols_info]
    print(columns)
    columns_str = ", ".join(f'"{col}"' for col in columns)
    print(columns_str)

    # Create a view that computes a random number once per row.
    con.execute("""
        CREATE OR REPLACE VIEW data_with_rand AS 
        SELECT *, random() AS rnd FROM data_view
    """)

    # Each COPY command now runs a subquery that computes a random number per row,
    # filters based on the desired ratio, and then selects only the original columns.
    train_query = f"SELECT {columns_str} FROM data_with_rand WHERE rnd < {train_ratio}"
    valid_query = f"SELECT {columns_str} FROM data_with_rand WHERE rnd >= {train_ratio} AND rnd < {train_ratio + valid_ratio}"
    test_query  = f"SELECT {columns_str} FROM data_with_rand WHERE rnd >= {train_ratio + valid_ratio}"


    # Use DuckDB's COPY command to export each split directly to JSON.
    con.execute(f"COPY ({train_query}) TO '{str(train_path)}' (FORMAT json)")
    con.execute(f"COPY ({valid_query}) TO '{str(valid_path)}' (FORMAT json)")
    con.execute(f"COPY ({test_query}) TO '{str(test_path)}' (FORMAT json)")

    con.close()

def split_jsonl_file_duckdb_parquet(
    input_path: Path,
    output_dir: Path,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed: int = None,
):
    """
    Splits a JSON Lines file into three files (train, valid, and test) by first converting
    the JSONL file into a Parquet file and then using DuckDB’s efficient columnar processing.
    
    Parameters:
        input_path (Path): Path to the input JSONL file.
        output_dir (Path): Directory where the split files will be saved.
        train_ratio (float): Fraction of data for training.
        valid_ratio (float): Fraction of data for validation.
        test_ratio (float): Fraction of data for testing.
        seed (int, optional): Random seed (ignored in DuckDB).
    """
    if seed is not None:
        print("Warning: DuckDB does not support setting a random seed; the provided seed will be ignored.")

    # Ensure input_path is absolute.
    if not input_path.is_absolute():
        input_path = (Path(__file__).parent / input_path).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    test_path = output_dir / "test.jsonl"

    # Connect to DuckDB (in-memory)
    con = duckdb.connect(database=":memory:")

    # Create a view on the JSONL file.
    con.execute(f"""
        CREATE OR REPLACE VIEW data_view AS 
        SELECT * FROM read_ndjson('{str(input_path.resolve())}', maximum_object_size = 40000000)
    """)

    # Convert the JSONL data to a Parquet file (temporary).
    parquet_file = output_dir / "temp_data.parquet"
    con.execute(f"""
        COPY (SELECT * FROM data_view) 
        TO '{str(parquet_file)}' (FORMAT PARQUET)
    """)

    # Create a view from the Parquet file.
    con.execute(f"""
        CREATE OR REPLACE VIEW data_parquet AS 
        SELECT * FROM read_parquet('{str(parquet_file)}')
    """)

    # Get the original column names from the Parquet data.
    cols_info = con.execute("PRAGMA table_info('data_parquet')").fetchall()
    columns = [row[1] for row in cols_info]
    columns_str = ", ".join(f'"{col}"' for col in columns)

    # Create a view that computes a random number once per row.
    con.execute("""
        CREATE OR REPLACE VIEW data_with_rand AS 
        SELECT *, random() AS rnd FROM data_parquet
    """)

    # Define queries for the three splits.
    train_query = f"SELECT {columns_str} FROM data_with_rand WHERE rnd < {train_ratio}"
    valid_query = f"SELECT {columns_str} FROM data_with_rand WHERE rnd >= {train_ratio} AND rnd < {train_ratio + valid_ratio}"
    test_query  = f"SELECT {columns_str} FROM data_with_rand WHERE rnd >= {train_ratio + valid_ratio}"

    # Export each split to a JSON Lines file.
    con.execute(f"COPY ({train_query}) TO '{str(train_path)}' (FORMAT json)")
    con.execute(f"COPY ({valid_query}) TO '{str(valid_path)}' (FORMAT json)")
    con.execute(f"COPY ({test_query}) TO '{str(test_path)}' (FORMAT json)")

    # Optionally remove the temporary Parquet file.
    parquet_file.unlink()

    con.close()

def split_jsonl_file_duckdb_parquet_chunked(
    input_path: Path,
    output_dir: Path,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    chunk_size=2000000,  # adjust based on available memory
    seed: int = None,
):
    """
    Splits a JSON Lines file into train/valid/test sets by first converting the JSONL file
    to Parquet in smaller chunks to avoid memory issues, and then splitting the data using DuckDB.
    
    Parameters:
        input_path (Path): Path to the input JSONL file.
        output_dir (Path): Directory where the split files will be saved.
        train_ratio (float): Fraction of data for training.
        valid_ratio (float): Fraction of data for validation.
        test_ratio (float): Fraction of data for testing.
        chunk_size (int): Number of rows per chunk to process.
        seed (int, optional): Random seed (ignored in DuckDB).
    """
    if seed is not None:
        print("Warning: DuckDB does not support setting a random seed; the provided seed will be ignored.")

    # Ensure absolute paths and output directory exists.
    if not input_path.is_absolute():
        input_path = (Path(__file__).parent / input_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary directory for chunked Parquet files.
    temp_parquet_dir = output_dir / "temp_parquet_chunks"
    temp_parquet_dir.mkdir(exist_ok=True)

    # Connect to DuckDB.
    con = duckdb.connect(database=str(output_dir / "temp_duckdb.db"))

    # Create a view on the JSONL file.
    con.execute(f"""
        CREATE OR REPLACE VIEW data_view AS 
        SELECT * FROM read_ndjson('{str(input_path.resolve())}', maximum_object_size = 200000000)
    """)

    # Determine the total number of rows.
    total_rows = con.execute("SELECT count(*) FROM data_view").fetchone()[0]
    print(f"Total rows in JSONL file: {total_rows}")

    # Process the file in chunks and write each chunk as a separate Parquet file.
    parquet_files = []
    for offset in range(0, total_rows, chunk_size):
        chunk_file = temp_parquet_dir / f"chunk_{offset}.parquet"
        print(f"Writing rows {offset} to {offset + chunk_size} to {chunk_file}")
        con.execute(f"""
            COPY (
                SELECT * FROM data_view 
                LIMIT {chunk_size} OFFSET {offset}
            ) TO '{str(chunk_file)}' (FORMAT PARQUET)
        """)
        parquet_files.append(str(chunk_file))

    # Create a view that reads from all the Parquet chunks.
    # DuckDB supports glob patterns for read_parquet, so we can combine them.
    glob_pattern = str(temp_parquet_dir / "chunk_*.parquet")
    con.execute(f"""
        CREATE OR REPLACE VIEW data_parquet AS 
        SELECT * FROM read_parquet('{glob_pattern}')
    """)

    # Retrieve column names.
    cols_info = con.execute("PRAGMA table_info('data_parquet')").fetchall()
    columns = [row[1] for row in cols_info]
    columns_str = ", ".join(f'"{col}"' for col in columns)

    # Create a view that computes a random number once per row.
    con.execute("""
        CREATE OR REPLACE VIEW data_with_rand AS 
        SELECT *, random() AS rnd FROM data_parquet
    """)

    # Define queries for train, valid, and test splits.
    train_query = f"SELECT {columns_str} FROM data_with_rand WHERE rnd < {train_ratio}"
    valid_query = f"SELECT {columns_str} FROM data_with_rand WHERE rnd >= {train_ratio} AND rnd < {train_ratio + valid_ratio}"
    test_query  = f"SELECT {columns_str} FROM data_with_rand WHERE rnd >= {train_ratio + valid_ratio}"

    # Define output file paths.
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    test_path  = output_dir / "test.jsonl"

    # Export the splits to JSON Lines.
    con.execute(f"COPY ({train_query}) TO '{str(train_path)}' (FORMAT json)")
    con.execute(f"COPY ({valid_query}) TO '{str(valid_path)}' (FORMAT json)")
    con.execute(f"COPY ({test_query}) TO '{str(test_path)}' (FORMAT json)")

    con.close()

    # Optionally, clean up temporary Parquet files.
    for file in temp_parquet_dir.iterdir():
        file.unlink()
    temp_parquet_dir.rmdir()

    print("Data splitting complete.")

if __name__ == "__main__":
    # Uncomment one group of lines at a time and run to test different methods.
    # This is to have a cold start for each run, since a MacBook Air has no active cooling. 

    # Define the input JSONL file (assumed to be in the same directory)
    input_file = (Path(__file__).parent / "book_reviews.0000.jsonl").resolve()       #  1.93GB JSON Lines file
    # input_file = (Path(__file__).parent / "book_technology.0000.jsonl").resolve()  #  4.34GB JSON Lines file
    # input_file = (Path(__file__).parent / "book_novel.0000.jsonl").resolve()       # 25.82GB JSON Lines file

    # --- Plain Python streaming approach ---
    # output_dir_plain = Path("output_splits_plain")
    # print("Running split_jsonl_file (plain Python streaming)...")
    # start_plain = time.perf_counter()
    # split_jsonl_file(
    #     input_path=input_file,
    #     output_dir=output_dir_plain,
    #     train_ratio=0.7,
    #     valid_ratio=0.15,
    #     test_ratio=0.15,
    #     sample_percentage=2,  # Use 100% of the data
    #     seed=42
    # )
    # end_plain = time.perf_counter()
    # print(f"split_jsonl_file took {end_plain - start_plain:.2f} seconds. Files saved in: {output_dir_plain}")

    # # --- DuckDB approach ---
    # output_dir_duckdb = Path("output_splits_duckdb")
    # print("Running split_jsonl_file_duckdb (DuckDB approach)...")
    # start_duckdb = time.perf_counter()
    # split_jsonl_file_duckdb(
    #     input_path=input_file,
    #     output_dir=output_dir_duckdb,
    #     train_ratio=0.7,
    #     valid_ratio=0.15,
    #     test_ratio=0.15,
    #     seed=42,         # Seed is ignored for DuckDB
    #     batch_size=1000  # Adjust the batch size as needed
    # )
    # end_duckdb = time.perf_counter()
    # print(f"split_jsonl_file_duckdb took {end_duckdb - start_duckdb:.2f} seconds. Files saved in: {output_dir_duckdb}")

    # # --- Improved DuckDB approach ---
    # output_dir_duckdb = Path("output_splits_duckdb_improved")
    # print("Running improved split_jsonl_file_duckdb (using DuckDB COPY)...")
    # start = time.perf_counter()
    # split_jsonl_file_duckdb_improved(
    #     input_path=input_file,
    #     output_dir=output_dir_duckdb,
    #     train_ratio=0.7,
    #     valid_ratio=0.15,
    #     test_ratio=0.15,
    #     seed=42  # This will be ignored
    # )
    # end = time.perf_counter()
    # print(f"split_jsonl_file_duckdb_improved took {end - start:.2f} seconds. Files saved in: {output_dir_duckdb}")

    # # --- Parquet DuckDB approach ---
    # output_dir_duckdb = Path("output_splits_duckdb_parquet")
    # print("Running Parquet-based split_jsonl_file_duckdb...")
    # start = time.perf_counter()
    # split_jsonl_file_duckdb_parquet(
    #     input_path=input_file,
    #     output_dir=output_dir_duckdb,
    #     train_ratio=0.7,
    #     valid_ratio=0.15,
    #     test_ratio=0.15,
    #     seed=42  # This will be ignored
    # )
    # end = time.perf_counter()
    # print(f"split_jsonl_file_duckdb_parquet took {end - start:.2f} seconds. Files saved in: {output_dir_duckdb}")

    # # --- Chunked Parquet DuckDB approach ---
    # output_dir_duckdb = Path("output_splits_duckdb_parquet_chunked")
    # print("Running Parquet-based chunked split_jsonl_file_duckdb_chunked...")
    # start = time.perf_counter()
    # split_jsonl_file_duckdb_parquet_chunked(
    #     input_path=input_file,
    #     output_dir=output_dir_duckdb,
    #     train_ratio=0.7,
    #     valid_ratio=0.15,
    #     test_ratio=0.15,
    #     seed=42  # This will be ignored
    # )
    # end = time.perf_counter()
    # print(f"split_jsonl_file_duckdb_parquet took {end - start:.2f} seconds. Files saved in: {output_dir_duckdb}")



    # Run a DuckDB query to check the average length of the string "text" field in the JSONL file.
    input_file = (Path(__file__).parent / "book_reviews.0000.jsonl").resolve()  #  original book_reviews.0000.jsonl,                    1.93GB JSON Lines file
    # input_file = (Path(__file__).parent / "train_1.1GB.jsonl").resolve()        #  50% split of the original book_reviews.0000.jsonl,   1.1GB JSON Lines file
    # input_file = (Path(__file__).parent / "train_100MB.jsonl").resolve()        #  10% split of the original book_reviews.0000.jsonl, 105.6MB JSON Lines file
    # input_file = (Path(__file__).parent / "train_20MB.jsonl").resolve()         #   2% split of the original book_reviews.0000.jsonl,  21.2MB JSON Lines file
    con = duckdb.connect(database=":memory:")
    con.execute(f"""
        CREATE OR REPLACE VIEW data_view AS
        SELECT * FROM read_ndjson('{str(input_file)}', maximum_object_size = 40000000)
    """)
    avg_length = con.execute("SELECT AVG(LENGTH(text)) FROM data_view").fetchone()[0]
    print(f"Average length of the 'text' field: {avg_length} characters")
    print(f"Total number of rows in the JSONL file: {con.execute('SELECT COUNT(*) FROM data_view').fetchone()[0]}")
    con.close()

    # Average the size, in tokens, of the text field in each line of the JSONL file. Should also the ammount of lines and print the minimum and maximum size in token.
    import tiktoken
    # enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.get_encoding("o200k_base")
    with input_file.open("r", encoding="utf-8") as infile:
        total_tokens = 0
        min_tokens = float('inf')
        max_tokens = float('-inf')
        count = 0
        for line in infile:
            data = json.loads(line)
            text_value = data.get("text", "")
            tokens = len(enc.encode(text_value))
            total_tokens += tokens
            min_tokens = min(min_tokens, tokens)
            max_tokens = max(max_tokens, tokens)
            count += 1
    avg_tokens = total_tokens / count if count > 0 else 0
    print(f"Average number of tokens: {avg_tokens}")
    print(f"Minimum number of tokens: {min_tokens}")
    print(f"Maximum number of tokens: {max_tokens}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Total number of lines: {count}")
    # Close the connection

    


    
