"""
Postprocessing for recommendation models, adapted for models
predicting from a mixed vocabulary (items and users).
"""

import numpy as np
import pandas as pd

def preds_to_item_recs_mixed_vocab(predictions_batches, item_id_reverse_map, num_items_in_vocab, top_k_items_to_return=10):
    
    print(f"Starting post-processing of prediction batches...")
    print(f"  Targeting top {top_k_items_to_return} item recommendations per user.")
    print(f"  Number of distinct items in vocabulary (for ID mapping): {num_items_in_vocab}")

    # List to accumulate recommendation data for all users before creating DataFrame
    user_recs_list = []

    # Process each batch of predictions
    for batch_idx, batch_output in enumerate(predictions_batches):
        original_user_ids_batch = batch_output['user_ids']
        # These are the model's top N predicted *entities* (items or users)
        predicted_entity_ids_batch = batch_output['preds']
        predicted_entity_scores_batch = batch_output['scores']

        print(f"  Processing batch {batch_idx + 1}/{len(predictions_batches)} with {len(original_user_ids_batch)} users.")

        # Process each user's predictions in the current batch
        for i in range(len(original_user_ids_batch)):
            user_id = original_user_ids_batch[i]
            entity_ids_for_user = predicted_entity_ids_batch[i]      # Shape: (num_entities_predicted_by_model,)
            entity_scores_for_user = predicted_entity_scores_batch[i] # Shape: (num_entities_predicted_by_model,)

            # Filter out non-item predictions:
            # Mapped item IDs are in the range [1, num_items_in_vocab].
            # Anything outside this (especially > num_items_in_vocab) is not an item.
            item_mask = (entity_ids_for_user >= 1) & (entity_ids_for_user <= num_items_in_vocab)
            
            actual_item_ids = entity_ids_for_user[item_mask]
            actual_item_scores = entity_scores_for_user[item_mask]

            # If no items were predicted for this user (e.g., all top entities were users), skip
            if len(actual_item_ids) == 0:
                continue

            # Sort the identified items by their scores in descending order
            # The model might have already sorted entities, but we need to re-sort among *filtered items*.
            sorted_indices = np.argsort(actual_item_scores)[::-1]
            
            # Select the top_k_items_to_return from the sorted items
            top_items_for_user = actual_item_ids[sorted_indices][:top_k_items_to_return]
            top_scores_for_user = actual_item_scores[sorted_indices][:top_k_items_to_return]
            
            # Add these recommendations to our list
            for item_idx in range(len(top_items_for_user)):
                user_recs_list.append({
                    'user_id': user_id,
                    'mapped_item_id': top_items_for_user[item_idx], # Store the mapped ID temporarily
                    'prediction_score': top_scores_for_user[item_idx]
                })

    # If no recommendations were generated at all
    if not user_recs_list:
        print("No valid item recommendations were generated after filtering.")
        return pd.DataFrame(columns=['user_id', 'item_id', 'prediction_score'])

    print(f"Collected {len(user_recs_list)} potential item recommendations across all users.")
    
    # Create a DataFrame from the collected recommendations
    recs_df = pd.DataFrame(user_recs_list)

    # Map the 'mapped_item_id' back to the 'original_item_id'
    if item_id_reverse_map is not None:
        print("Mapping item IDs back to original item IDs...")
        recs_df['item_id'] = recs_df['mapped_item_id'].map(item_id_reverse_map)
        
        # Handle cases where a mapped_item_id might not be in the reverse map.
        # This should be rare if num_items_in_vocab and the item_mask are correct.
        # Rows with NaN item_id after mapping are dropped as they don't correspond to known items.
        original_rows = len(recs_df)
        recs_df.dropna(subset=['item_id'], inplace=True)
        dropped_rows = original_rows - len(recs_df)
        if dropped_rows > 0:
            print(f"  Dropped {dropped_rows} recommendations due to missing original item ID after mapping.")
        
        # Ensure item_id is integer type if it became float due to NaNs before dropna
        if 'item_id' in recs_df.columns and not recs_df['item_id'].empty:
             recs_df['item_id'] = recs_df['item_id'].astype(recs_df['mapped_item_id'].infer_objects().dtype) # Keep original type if possible
    else:
        # If no reverse map is provided, assume mapped_item_id is the final item ID.
        print("No item_id_reverse_map provided. Using mapped_item_id as item_id.")
        recs_df['item_id'] = recs_df['mapped_item_id']


    # Ensure essential columns are present, even if no valid items were mapped
    if 'item_id' not in recs_df.columns:
        recs_df['item_id'] = np.nan # Add empty column if it's missing

    # Select and order final columns
    final_df = recs_df[['user_id', 'item_id', 'prediction_score']].copy()
    
    # Sort the final recommendations
    final_df.sort_values(['user_id', 'prediction_score'], ascending=[True, False], inplace=True)
    
    print(f"Post-processing complete. Returning {len(final_df)} item recommendations.")
    return final_df


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # This example demonstrates how to use the function.
    # In a real scenario, `predictions_batches` would come from your model's output,
    # and `item_id_reverse_map` and `num_items_in_vocab` from your preprocessing step.

    # --- Setup (mimicking outputs from preprocessing) ---
    # Example: item_map from preprocessing was {'itemA': 1, 'itemB': 2, 'itemC': 3}
    # Example: user_map from preprocessing was {'userX': 4, 'userY': 5}
    example_item_map = {'orig_item_1': 1, 'orig_item_2': 2, 'orig_item_3': 3, 'orig_item_4': 4}
    num_items = len(example_item_map) # This is num_items_in_vocab
    
    # Create the reverse map needed for post-processing
    example_item_id_reverse_map = {v: k for k, v in example_item_map.items()}
    
    # Mapped ID for a user (example: user1 maps to 5)
    mapped_user1_id = num_items + 1 # 5
    mapped_user2_id = num_items + 2 # 6


    # --- Mock Model Predictions ---
    # Batch 1
    preds_batch_1 = {
        'user_ids': np.array(['user101', 'user102']), # Original user IDs
        'preds': np.array([ # Mapped entity IDs predicted by model
            [1, mapped_user1_id, 2, 3],  # User101: item1, user_entity, item2, item3
            [mapped_user2_id, 4, 1, mapped_user1_id]   # User102: user_entity, item4, item1, user_entity
        ]),
        'scores': np.array([ # Corresponding scores
            [0.9, 0.85, 0.8, 0.7],
            [0.95, 0.9, 0.8, 0.75]
        ])
    }
    # Batch 2
    preds_batch_2 = {
        'user_ids': np.array(['user103']),
        'preds': np.array([
            [mapped_user1_id, mapped_user2_id, mapped_user1_id, mapped_user2_id] # Only user entities predicted
        ]),
        'scores': np.array([
            [0.9, 0.8, 0.7, 0.6]
        ])
    }
    
    mock_predictions_batches = [preds_batch_1, preds_batch_2]
    
    # --- Run Post-processing ---
    recommendations_df = preds_to_item_recs_mixed_vocab(
        predictions_batches=mock_predictions_batches,
        item_id_reverse_map=example_item_id_reverse_map,
        num_items_in_vocab=num_items,
        top_k_items_to_return=2 # Get top 2 item recommendations
    )

    print("\n--- Generated Recommendations ---")
    print(recommendations_df)

