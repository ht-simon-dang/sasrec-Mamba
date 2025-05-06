import models

if __name__ == '__main__':
    # --- Configuration ---
    VOCAB_SIZE = 10000 + 1 # Example: 10k items + 1 padding token
    PADDING_IDX = 0
    MAX_LEN = 50 # Example max sequence length
    D_MODEL = 128 # Embedding and hidden size
    N_LAYER = 4   # Number of Mamba layers
    D_STATE = 16  # SSM state expansion factor
    D_CONV = 4    # Local convolution width
    EXPAND = 2    # Block expansion factor

    # Mamba configuration dictionary
    mamba_config = {
        'd_model': D_MODEL,
        'n_layer': N_LAYER,
        'd_state': D_STATE,
        'd_conv': D_CONV,
        'expand': EXPAND,
        # Add other necessary Mamba parameters here if needed
        # e.g., 'bias': False, 'conv_bias': True
    }

    # --- Model Instantiation ---
    model = MAMBA4Rec(
        vocab_size=VOCAB_SIZE,
        mamba_config=mamba_config,
        add_head=True,
        tie_weights=True,
        padding_idx=PADDING_IDX,
        init_std=0.02
    )

    print("MAMBA4Rec Model:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params:,}")


    # --- Dummy Input ---
    batch_size = 4
    # Create sequences of varying lengths, padded to MAX_LEN
    seq_lengths = torch.randint(5, MAX_LEN + 1, (batch_size,))
    dummy_input_ids = torch.full((batch_size, MAX_LEN), PADDING_IDX, dtype=torch.long)
    for i in range(batch_size):
        seq_len = seq_lengths[i]
        dummy_input_ids[i, :seq_len] = torch.randint(1, VOCAB_SIZE, (seq_len,))

    # --- Forward Pass ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dummy_input_ids = dummy_input_ids.to(device)
        print(f"\nRunning on device: {device}")

        with torch.no_grad(): # No need to compute gradients for this example
            outputs = model(dummy_input_ids)

        print(f"\nInput shape: {dummy_input_ids.shape}")
        print(f"Output shape: {outputs.shape}") # Should be (batch_size, MAX_LEN, VOCAB_SIZE)

    except Exception as e:
        print(f"\nError during forward pass: {e}")
        print("This might happen if 'mamba-ssm' is not installed correctly or if using the placeholder.")

