import numpy as np
import torch
from typing import Dict, Optional, Any, Union

class ActionTokenizer:
    """Utility class for converting between continuous actions and discrete tokens."""
    
    def __init__(
        self, 
        vocab_size: int,
        n_action_bins: int,
        norm_stats: Dict[str, Dict[str, Any]],
        pad_to_multiple_of: int = 0
    ):
        """Initialize the ActionTokenizer.
        
        Args:
            vocab_size: Size of the vocabulary (before padding)
            n_action_bins: Number of bins to discretize actions
            norm_stats: Dictionary of normalization statistics per dataset
            pad_to_multiple_of: Optional padding added to vocab size
        """
        self.n_action_bins = n_action_bins
        self.norm_stats = norm_stats
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # Set effective vocab size (before padding)
        self.vocab_size = vocab_size
        
        # Compute action bins
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def _check_unnorm_key(self, unnorm_key: Optional[str] = None) -> str:
        """Validate and get the correct unnormalization key."""
        if unnorm_key is None:
            assert len(self.norm_stats) == 1, (
                f"Multiple datasets available, please pass a `unnorm_key` from: {self.norm_stats.keys()}"
            )
            unnorm_key = next(iter(self.norm_stats.keys()))

        assert unnorm_key in self.norm_stats, (
            f"Invalid unnorm_key, please choose from: {self.norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get normalization statistics for the specified dataset."""
        unnorm_key = self._check_unnorm_key(unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the action space."""
        unnorm_key = self._check_unnorm_key(unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def actions_to_tokens(
        self, 
        actions: Union[np.ndarray, torch.Tensor], 
        unnorm_key: Optional[str] = None
    ) -> torch.LongTensor:
        """Convert continuous actions to token IDs."""
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
            
        # Get normalization stats
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        
        # Normalize actions to [-1, 1] range using the exact inverse of the unnormalization formula
        normalized_actions = np.where(
            mask,
            2 * ((actions - action_low) / (action_high - action_low)) - 1,
            actions
        )
        normalized_actions = np.clip(normalized_actions, -1, 1)
        
        # Find closest bin center
        discretized_actions = np.array([np.abs(self.bin_centers - x).argmin() for x in normalized_actions.flat]).reshape(normalized_actions.shape)
        
        # Convert to token IDs using same formula as in predict_action
        token_ids = self.vocab_size - discretized_actions - 1
        
        return torch.from_numpy(token_ids).long()

    def tokens_to_actions(
        self, 
        token_ids: Union[np.ndarray, torch.LongTensor],
        unnorm_key: Optional[str] = None
    ) -> np.ndarray:
        """Convert token IDs back to continuous actions."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        # Convert tokens to discretized actions
        discretized_actions = self.vocab_size - token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        
        # Convert to normalized actions using bin centers
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            ((normalized_actions + 1) * (action_high - action_low)) / 2 + action_low,
            normalized_actions,
        )

        return actions
