"""
Process CodeComplex XLSX Dataset
Reads Excel file and extracts features for training
"""

import pickle
from collections import Counter
import pandas as pd
from code_parser import CodeParser


def process_codecomplex_dataset(
    input_file='data/codecomplex_8000_parser_ready.xlsx',
    output_pkl='data/processed_data.pkl',
    output_json='data/training_ready.json',
    max_samples=None,
    complexity_filter=None
):
    """
    Process CodeComplex XLSX dataset
    """

    print("=" * 60)
    print("PROCESSING CODECOMPLEX DATASET (XLSX)")
    print("=" * 60)

    parser = CodeParser()
    processed_samples = []
    complexity_counts = Counter()

    error_count = 0

    print(f"\nReading from: {input_file}")
    print("Loading Excel file...")

    try:
        print("Reading XLSX file (this may take a moment for large files)...")
        df = pd.read_excel(input_file, engine='openpyxl')

        # Strip column whitespace (safety)
        df.columns = df.columns.str.strip()

        total_rows = len(df)
        print(f"✓ Loaded {total_rows} rows from Excel")

        print(f"\nAvailable columns: {list(df.columns)}")

        code_column = 'src'
        complexity_column = 'complexity'

        print(f"\nProcessing rows...")
        print(f"Code column: '{code_column}'")
        print(f"Complexity column: '{complexity_column}'")

        python_count = 0

        for idx, row in df.iterrows():

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{total_rows} rows... ({python_count} valid samples)")

            if max_samples and python_count >= max_samples:
                print(f"\n✓ Reached max_samples limit: {max_samples}")
                break

            try:
                code = str(row[code_column]) if pd.notna(row[code_column]) else ''
                complexity = str(row[complexity_column]) if pd.notna(row[complexity_column]) else ''

                if not code or code == 'nan' or not complexity or complexity == 'nan':
                    error_count += 1
                    continue

                if complexity_filter and complexity not in complexity_filter:
                    continue

                complexity = normalize_complexity(complexity)

                if not complexity:
                    error_count += 1
                    continue

                try:
                    features = parser.extract_features(code)

                    sample = {
                        'code': code,
                        'complexity': complexity,
                        'features': features
                    }

                    processed_samples.append(sample)
                    complexity_counts[complexity] += 1
                    python_count += 1

                except SyntaxError:
                    error_count += 1
                    continue
                except Exception:
                    error_count += 1
                    continue

            except Exception:
                error_count += 1
                continue

        print(f"\n✓ Read {total_rows} rows from Excel")
        print(f"✓ Extracted {python_count} valid samples")
        print(f"✗ Skipped {error_count} errors/invalid entries")

        print(f"\nSaving processed data...")

        with open(output_pkl, 'wb') as f:
            pickle.dump(processed_samples, f)
        print(f"✓ Saved to {output_pkl}")

        import json
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(processed_samples, f, indent=2)
        print(f"✓ Saved to {output_json}")

        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(processed_samples)}")

        print("\nComplexity Distribution:")
        for complexity, count in sorted(complexity_counts.items()):
            percentage = (count / len(processed_samples)) * 100
            print(f"  {complexity:12s}: {count:5d} samples ({percentage:5.1f}%)")

        print("\nFeature Statistics:")
        if processed_samples:
            sample_features = processed_samples[0]['features']
            print(f"  Features per sample: {len(sample_features)}")
            print(f"  Feature names: {list(sample_features.keys())}")

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"\nNext step: Run 'python train_model.py' to train the model")

        return processed_samples

    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found: {input_file}")
        return []

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def normalize_complexity(complexity_str):

    if not complexity_str:
        return None

    complexity = complexity_str.strip().lower()

    mappings = {

        # already normalized Big-O
        'o(1)': 'O(1)',
        'o(log n)': 'O(log n)',
        'o(n)': 'O(n)',
        'o(n log n)': 'O(n log n)',
        'o(n^2)': 'O(n^2)',
        'o(n^3)': 'O(n^3)',

        # dataset words
        'constant': 'O(1)',
        'logarithmic': 'O(log n)',
        'linear': 'O(n)',
        'linearithmic': 'O(n log n)',
        'quadratic': 'O(n^2)',
        'cubic': 'O(n^3)'
    }

    return mappings.get(complexity, None)



if __name__ == '__main__':

    INPUT_FILE = 'data/codecomplex_8000_parser_ready.xlsx'
    OUTPUT_PKL = 'data/processed_data.pkl'
    OUTPUT_JSON = 'data/training_ready.json'

    MAX_SAMPLES = None
    COMPLEXITY_FILTER = None

    processed = process_codecomplex_dataset(
        input_file=INPUT_FILE,
        output_pkl=OUTPUT_PKL,
        output_json=OUTPUT_JSON,
        max_samples=MAX_SAMPLES,
        complexity_filter=COMPLEXITY_FILTER
    )

    print(f"\n✅ Dataset ready for training!")
    print(f"   Samples: {len(processed)}")
    print(f"   File: {OUTPUT_PKL}")
