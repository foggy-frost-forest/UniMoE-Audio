import torch


def compress_matrix(A: torch.Tensor, mask: torch.Tensor, force_dim: int = None, allow_larger_dim=None) -> torch.Tensor:
    """
    Compresses matrix A (S, E, ...) based on the mask (S, E).

    Args:
        A (torch.Tensor): The input matrix with shape (S, E, ...).
        mask (torch.Tensor): The binary mask matrix with shape (S, E).
        force_dim (int, optional): If provided, forces the first dimension of the output B to this value.
                                   Otherwise, it's determined by the max number of 1s in any mask column.
        allow_larger_dim (bool, optional):
            - If force_dim causes the target dimension to be > S (original rows):
                - True: Allows padding B with zeros.
                - False: Raises an AssertionError.
                - None (default): Allows padding with zeros and prints a warning.

    Returns:
        torch.Tensor: The compressed matrix B with shape (X_target_dim, E, ...).
    """
    if A.shape[:2] != mask.shape:
        raise ValueError("First two dimensions of A and mask must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError(
            f"mask must only contain 0s and 1s. dtype: {mask.dtype}. "
            f"Invalid elements found at indices: {((mask != 0) & (mask != 1)).nonzero().tolist()} "  # Get indices of elements not 0 AND not 1
            f"with corresponding values: {mask[((mask != 0) & (mask != 1))].tolist()}. "  # Get the values at those indices
            f"\nOriginal mask (showing up to first 20 elements if large):\n{mask.flatten()[:20]}{'...' if mask.numel() > 20 else ''}"
        )

    S, E = mask.shape
    trailing_dims_shape = A.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = A.device

    ones_per_column = mask.sum(dim=0)
    X = ones_per_column.max().item() if force_dim is None else force_dim

    if X == 0:
        return torch.empty((0, E, *trailing_dims_shape), dtype=A.dtype, device=device)

    # sorted_row_indices[r, c] gives the original row index in A
    # that moves to the r-th position in the sorted version of column c.
    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True)  # Shape (S, E)

    # Expand sorted_row_indices_2d to match A's dimensions for gather
    # Shape: (S, E, 1, 1, ...) -> (S, E, D1, D2, ...)
    view_shape_for_indices = (S, E, *((1,) * num_trailing_dims))
    expanded_indices = sorted_row_indices_2d.view(view_shape_for_indices).expand_as(A)

    # Gather elements from A
    A_gathered = torch.gather(A, 0, expanded_indices)  # Shape (S, E, ...)

    # Take the top X rows
    if X <= A_gathered.shape[0]:
        B_candidate = A_gathered[:X, ...]  # Shape (X, E, ...)
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
            print(f"[Warning compress_matrix] Target dimension X ({X}) is larger than "
                      f"A's original row count S ({S}). Padding B_candidate with zeros.")
        B_candidate = A_gathered  # Shape (X, E, ...)
        zeros_shape = [X - A_gathered.shape[0]] + list(B_candidate.shape[1:])
        B_candidate = torch.cat((B_candidate, torch.zeros(zeros_shape, dtype=B_candidate.dtype, device=B_candidate.device)), dim=0)  # Shape (X_target_dim, E, ...)
    else:
        raise AssertionError(
                f"Target dimension X ({X}) is larger than A's original row count S ({S}) "
                f"and allow_larger_dim is False. Padding is disallowed."
            )

    # Create a mask for B to zero out padding
    row_indices_for_B = torch.arange(X, device=device).unsqueeze(1)  # Shape (X, 1)
    b_mask_2d = row_indices_for_B < ones_per_column.unsqueeze(0)  # Shape (X, E)

    # Expand b_mask_2d and apply it
    # Shape: (X, E, 1, 1, ...) -> (X, E, D1, D2, ...)
    view_shape_for_b_mask = (X, E, *((1,) * num_trailing_dims))
    # B = torch.zeros_like(B_candidate) # Initialize B
    # expanded_b_mask_for_B = b_mask_2d.view(view_shape_for_b_mask).expand_as(B_candidate)
    # B[expanded_b_mask_for_B] = B_candidate[expanded_b_mask_for_B]
    # More concise way:
    B = B_candidate * b_mask_2d.view(view_shape_for_b_mask).to(A.dtype)

    return B


def decompress_matrix(B: torch.Tensor, mask: torch.Tensor, allow_larger_dim=None) -> torch.Tensor:
    """
    Decompresses matrix B (X, E, ...) back to original shape (S, E, ...) using mask (S, E).

    Args:
        B (torch.Tensor): The compressed matrix with shape (X, E, ...).
        mask (torch.Tensor): The original binary mask matrix with shape (S, E).
        allow_larger_dim (bool, optional):
            - If B.shape[0] (input X) > S (target rows for A):
                - True: Allows truncating B to S rows.
                - False: Raises an AssertionError.
                - None (default): Allows truncating B to S rows and prints a warning.
    Returns:
        torch.Tensor: The decompressed matrix A_reconstructed with shape (S, E, ...).
    """
    if B.shape[1] != mask.shape[1]:
        raise ValueError("B's second dimension and mask's second dimension (E) must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all(): # Simplified error for brevity here, use your detailed one
        raise ValueError("mask must only contain 0s and 1s.")

    S, E = mask.shape
    X = B.shape[0]
    trailing_dims_shape = B.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = B.device

    if X == 0:  # If B is empty (e.g., mask was all zeros)
        return torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)

    if X <= S:
        pass
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
                print(f"[Warning decompress_matrix] Input B.shape[0] ({X}) is larger than "
                      f"target A's row count S ({S}). Truncating B to its first {S} rows.")
        B = B[:S, ...]
        X = S
    else:
        raise AssertionError(
                f"Input B.shape[0] ({X}) is larger than target A's row count S ({S}) "
                f"and allow_larger_dim is False. Truncation is disallowed."
            )

    # Reconstruct sorted_row_indices as in compression
    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True)  # Shape (S, E)

    # These are the row indices in A where elements of B should be placed.
    target_A_row_indices_2d = sorted_row_indices_2d[:X, :]  # Shape (X, E)

    # Initialize A_reconstructed with zeros
    A_reconstructed = torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)

    # Expand target_A_row_indices_2d to match B's dimensions for scatter_
    # Shape: (X, E, 1, 1, ...) -> (X, E, D1, D2, ...)
    view_shape_for_target_indices = (X, E, *((1,) * num_trailing_dims))
    expanded_target_indices = target_A_row_indices_2d.view(view_shape_for_target_indices).expand_as(B)

    # Scatter elements from B into A_reconstructed
    A_reconstructed.scatter_(dim=0, index=expanded_target_indices, src=B)

    # Optional: Explicitly ensure positions where mask is 0 are zero.
    # This should be redundant if B was formed correctly by compress_matrix
    # and scatter_ works as intended.
    # expanded_mask_for_A = mask.view(S, E, *((1,)*num_trailing_dims)).expand_as(A_reconstructed)
    # A_reconstructed = A_reconstructed * expanded_mask_for_A.to(A_reconstructed.dtype)

    return A_reconstructed

