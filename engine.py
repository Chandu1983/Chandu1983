# ==============================================================================
# GADE UNIVERSAL DIMENSIONLESS BOUNDARY METRIC (CORE MATHEMATICAL ENGINE)
# Framework Construction: Pi_I = Pi_B - Pi_T
# Lead Inventor: Chandrakant Shivram Gade, Nashik, Maharashtra, India.
# Version: 5.0.0 | Multi-Star Production Blueprint
# ==============================================================================

import numpy as np
import pandas as pd

class SNAPBayesianEngine:
    def __init__(self, alpha=0.3):
        """
        Initializes the Gade Dimensionless Core Engine components.
        """
        self.alpha = alpha
        self.eps = 1e-12

    def flux(self, m):
        """Converts astronomical magnitudes to pure radiation flux vector."""
        return 10 ** (-0.4 * np.asarray(m, dtype=float))

    def normalize(self, x):
        """Normalizes the fluid flux relative to its dynamic baseline mean."""
        return np.asarray(x, dtype=float) / (np.mean(x) + self.eps)

    def compute_state(self, m):
        """
        Computes Pi_B (Driving Force) and Pi_T (Resistance Tension).
        Implements the mandatory 101-day rolling window to destroy fake noise.
        """
        F = self.flux(m)
        
        # Core 101-day dynamic rolling filter to wipe out atmospheric turbulence
        if len(F) > 50:
            F = pd.Series(F).rolling(window=101, center=True, min_periods=1).mean().to_numpy()
        
        # Pi_B represents normalized core driving fluid pressure matrix
        B = self.normalize(F)
        
        # Pi_T captures active boundary structural tension / skin friction drag
        grad_B = np.gradient(B) if len(B) >= 2 else np.zeros_like(B)
        T = 1.0 + self.alpha * grad_B
        return B, T

    def score(self, m):
        """
        Calculates the Gade Instability Index (Pi_I = Pi_B - Pi_T) 
        and maps it directly to a Sigmoid Probability curve.
        """
        B, T = self.compute_state(m)
        I = B - T  # Pure Dimensionless Inversion State Ledger
        
        if len(I) < 2:
            return B, T, I, np.full_like(I, 0.5)
            
        # Core dynamic stability vector tracking
        grad_I = np.gradient(I)
        std_I = np.array([np.std(I[max(0, i-50):i+1]) for i in range(len(I))])
        
        z = 2.5 * I + 1.5 * grad_I + 0.2 * std_I
        P = 1.0 / (1.0 + np.exp(-z))  # Sigmoid Mapping Layer Execution
        return B, T, I, P

def filter_gade_events(P, threshold, window=1200):
    """
    Implements the non-maximum local peak suppression filter.
    Locks true dynamic peaks within a strict 1200-point epoch window boundary.
    """
    P = np.asarray(P, dtype=float)
    raw_indices = np.where(P > threshold)[0]
    filtered = []
    
    for idx in raw_indices:
        w_start = max(0, idx - window)
        w_end = min(len(P), idx + window + 1)
        if P[idx] == np.max(P[w_start:w_end]):
            if int(idx) not in filtered:
                filtered.append(int(idx))
    return filtered
