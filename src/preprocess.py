
import pandas as pd
import numpy as np
from collections import defaultdict

def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """
    Add user_id instead here.
    """
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df

def filter_items(df, item_min_count, item_col='item_id'):

    print('Filtering items..')

    item_count = df.groupby(item_col).user_id.nunique()

    item_ids = item_count[item_count >= item_min_count].index
    print(f'Number of items before {len(item_count)}')
    print(f'Number of items after {len(item_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df.item_id.isin(item_ids)]
    print(f'Interactions length after: {len(df)}')

    return df

def filter_users(df, user_min_count, user_col='user_id'):

    print('Filtering users..')

    user_count = df.groupby(user_col).item_id.nunique()

    user_ids = user_count[user_count >= user_min_count].index
    print(f'Number of users before {len(user_count)}')
    print(f'Number of users after {len(user_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df.user_id.isin(user_ids)]
    print(f'Interactions length after: {len(df)}')

    return df

## Empirial by forgetting, use user_id to avoid forget.
## Enable DyMap, by original length
def map_ids(df, user_col='user_id', item_col='item_id'):
    """
    item_count = 3417
    user_id = user_id + 3417
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
    Replace add_time_idx;

    i0 i1 u1, 
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

        user_sequences[user_id] = new_sequence
        processed_users += 1
        if processed_users % 1000 == 0:
             print(f"  Processed {processed_users}/{total_users} users...")

    print(f"Finished generating sequences for {processed_users} users.")
    return dict(user_sequences)
