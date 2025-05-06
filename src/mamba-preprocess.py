"""
Pre-processing script for recommendation data, including filtering
and sequence generation with user ID injection.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# --- Filtering Functions (from original script) ---

def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """Add time index (0-based) to interactions dataframe."""
    if sort:
        print(f"Sorting interactions by {user_col} and {timestamp_col}...")
        df = df.sort_values([user_col, timestamp_col])

    print("Adding time indices (time_idx and time_idx_reversed)...")
    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)
    print("Time indices added.")
    return df

def filter_items(df, item_min_count, user_col='user_id', item_col='item_id'):
    """Filter out items with fewer than item_min_count interactions."""
    print(f"Filtering items with less than {item_min_count} interactions...")
    item_counts = df.groupby(item_col)[user_col].nunique() # Count unique users per item
    valid_items = item_counts[item_counts >= item_min_count].index
    n_items_before = df[item_col].nunique()
    n_interactions_before = len(df)

    df_filtered = df[df[item_col].isin(valid_items)].copy() # Use .copy() to avoid SettingWithCopyWarning

    n_items_after = df_filtered[item_col].nunique()
    n_interactions_after = len(df_filtered)

    print(f"  Items before: {n_items_before}")
    print(f"  Items after: {n_items_after} (removed {n_items_before - n_items_after})")
    print(f"  Interactions before: {n_interactions_before}")
    print(f"  Interactions after: {n_interactions_after} (removed {n_interactions_before - n_interactions_after})")
    print("Item filtering done.")
    return df_filtered

def filter_users(df, user_min_count, user_col='user_id', item_col='item_id'):
    """Filter out users with fewer than user_min_count interactions."""
    print(f"Filtering users with less than {user_min_count} interactions...")
    user_counts = df.groupby(user_col)[item_col].nunique() # Count unique items per user
    valid_users = user_counts[user_counts >= user_min_count].index
    n_users_before = df[user_col].nunique()
    n_interactions_before = len(df)

    df_filtered = df[df[user_col].isin(valid_users)].copy() # Use .copy() to avoid SettingWithCopyWarning

    n_users_after = df_filtered[user_col].nunique()
    n_interactions_after = len(df_filtered)

    print(f"  Users before: {n_users_before}")
    print(f"  Users after: {n_users_after} (removed {n_users_before - n_users_after})")
    print(f"  Interactions before: {n_interactions_before}")
    print(f"  Interactions after: {n_interactions_after} (removed {n_interactions_before - n_interactions_after})")
    print("User filtering done.")
    return df_filtered

# --- New Functions for ID Mapping and Sequence Generation ---

def map_ids(df, user_col='user_id', item_col='item_id'):
    """
    Maps original user_ids and item_ids to contiguous integer ranges
    suitable for embedding layers. Item IDs start from 1 (0 is often
    reserved for padding). User IDs start after the last item ID.

    Returns:
        df (pd.DataFrame): DataFrame with new 'mapped_user_id' and 'mapped_item_id' columns.
        user_map (dict): Mapping from original user_id to mapped_user_id.
        item_map (dict): Mapping from original item_id to mapped_item_id.
        num_items (int): Number of unique items (determines the start index for users).
        num_users (int): Number of unique users.
        total_entities (int): Total number of unique items + users in the mapped space.
    """
    print("Mapping original IDs to contiguous integer ranges...")

    # Map Items: Start from 1 (0 reserved for padding)
    unique_items = df[item_col].unique()
    item_map = {item_id: i + 1 for i, item_id in enumerate(unique_items)}
    num_items = len(item_map)
    df['mapped_item_id'] = df[item_col].map(item_map)
    print(f"  Mapped {num_items} unique items (IDs 1 to {num_items}).")

    # Map Users: Start after the last item ID
    unique_users = df[user_col].unique()
    user_map = {user_id: i + num_items + 1 for i, user_id in enumerate(unique_users)}
    num_users = len(user_map)
    df['mapped_user_id'] = df[user_col].map(user_map)
    print(f"  Mapped {num_users} unique users (IDs {num_items + 1} to {num_items + num_users}).")

    total_entities = num_items + num_users + 1 # +1 because IDs start from 1

    print(f"  Total unique entities in mapped space: {total_entities - 1} (Range: 1 to {total_entities - 1})")
    print("ID mapping done.")
    return df, user_map, item_map, num_items, num_users, total_entities


def create_sequences_with_user_id(df, user_col='user_id', mapped_item_col='mapped_item_id', mapped_user_col='mapped_user_id'):
    """
    Generates sequences for each user, inserting the user's mapped ID
    after every two mapped item IDs.

    Args:
        df (pd.DataFrame): DataFrame MUST be sorted by user and timestamp,
                           and contain mapped IDs.
        user_col (str): Column name for original user ID (used for grouping).
        mapped_item_col (str): Column name for mapped item IDs.
        mapped_user_col (str): Column name for mapped user IDs.

    Returns:
        dict: A dictionary where keys are original user IDs and values are
              lists representing the generated sequences (containing mixed
              mapped item IDs and mapped user IDs).
    """
    print("Generating sequences with user ID injection...")
    user_sequences = defaultdict(list)
    processed_users = 0

    # Group by the original user ID to process each user's history
    grouped = df.groupby(user_col)
    total_users = len(grouped)

    for user_id, group in grouped:
        # Get the sequence of mapped item IDs for this user
        item_sequence = group[mapped_item_col].tolist()
        # Get the single mapped user ID for this user (it's the same for all rows in the group)
        mapped_user_id = group[mapped_user_col].iloc[0]

        new_sequence = []
        item_count_since_last_user = 0

        for item_id in item_sequence:
            new_sequence.append(item_id)
            item_count_since_last_user += 1

            # Insert user ID after every 2 items
            if item_count_since_last_user == 2:
                new_sequence.append(mapped_user_id)
                item_count_since_last_user = 0 # Reset counter

        # Optional: Handle the case where the last segment has only one item.
        # Depending on the modeling goal, you might want to append the user ID
        # anyway, or leave it as is. Current logic leaves it as is.
        # Example: [item1, item2, user, item3] -> remains as is.
        # If you want to always end with user ID if there are remaining items:
        # if item_count_since_last_user > 0:
        #    new_sequence.append(mapped_user_id)

        user_sequences[user_id] = new_sequence
        processed_users += 1
        if processed_users % 1000 == 0:
             print(f"  Processed {processed_users}/{total_users} users...")

    print(f"Finished generating sequences for {processed_users} users.")
    return dict(user_sequences)


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create a dummy DataFrame
    data = {
        'user_id': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'D'],
        'item_id': [101, 102, 103, 104, 105, 201, 202, 101, 301, 102, 401],
        'timestamp': [1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 1] # Timestamps ensure order
    }
    interactions_df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(interactions_df)
    print("-" * 30)

    # 2. Define filtering parameters (adjust as needed)
    ITEM_MIN_COUNT = 2 # Keep items interacted with by at least 2 users
    USER_MIN_COUNT = 3 # Keep users who interacted with at least 3 unique items

    # 3. Apply filtering and sorting
    interactions_df = add_time_idx(interactions_df) # Sorts by user, timestamp
    interactions_df = filter_items(interactions_df, ITEM_MIN_COUNT)
    interactions_df = filter_users(interactions_df, USER_MIN_COUNT)

    print("\nDataFrame after filtering and sorting:")
    print(interactions_df)
    print("-" * 30)

    # Check if DataFrame is empty after filtering
    if interactions_df.empty:
        print("DataFrame is empty after filtering. No sequences to generate.")
    else:
        # 4. Map IDs
        mapped_df, user_map, item_map, num_items, num_users, total_entities = map_ids(interactions_df)
        print("\nDataFrame with Mapped IDs:")
        print(mapped_df)
        print("\nUser Map:", user_map)
        print("Item Map:", item_map)
        print(f"Num Items: {num_items}, Num Users: {num_users}, Total Entities: {total_entities}")
        print("-" * 30)

        # 5. Generate sequences with user ID injection
        # Ensure sorting again just in case (although add_time_idx should handle it)
        mapped_df = mapped_df.sort_values(['user_id', 'timestamp'])
        final_sequences = create_sequences_with_user_id(mapped_df)

        print("\nGenerated Sequences (Original User ID -> Mixed Mapped Sequence):")
        for user, seq in final_sequences.items():
            print(f"User '{user}': {seq}")

        # Example: How to interpret the sequence for User 'A'
        # Original items for A after filtering: [101, 102]
        # Mapped items: [item_map[101], item_map[102]]
        # Mapped user: user_map['A']
        # Expected sequence: [mapped_item_101, mapped_item_102, mapped_user_A]
        print("\n--- Example Interpretation ---")
        if 'A' in final_sequences:
             print(f"User 'A' original items (post-filter): {interactions_df[interactions_df['user_id'] == 'A']['item_id'].tolist()}")
             print(f"User 'A' mapped user ID: {user_map.get('A')}")
             print(f"User 'A' mapped item IDs: {[item_map.get(i) for i in interactions_df[interactions_df['user_id'] == 'A']['item_id'].tolist()]}")
             print(f"User 'A' generated sequence: {final_sequences['A']}")


