"""
Balance Processed Dataset
Equalizes all complexity classes before training
"""

import pickle
from collections import defaultdict
import random


def balance_dataset():

    print("=" * 60)
    print("BALANCING DATASET")
    print("=" * 60)

    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"\nOriginal samples: {len(data)}")

    class_map = defaultdict(list)

    for sample in data:
        class_map[sample['complexity']].append(sample)

    print("\nOriginal Distribution:")
    for k, v in class_map.items():
        print(f"{k:12s} : {len(v)}")

    # find smallest class
    min_count = min(len(v) for v in class_map.values())

    print(f"\nTarget samples per class: {min_count}")

    balanced = []

    for k, v in class_map.items():
        sampled = random.sample(v, min_count)
        balanced.extend(sampled)

    random.shuffle(balanced)

    print(f"\nBalanced samples: {len(balanced)}")

    with open('data/balanced_data.pkl', 'wb') as f:
        pickle.dump(balanced, f)

    print("\n✔ Balanced dataset saved as balanced_data.pkl")


if __name__ == "__main__":
    balance_dataset()
