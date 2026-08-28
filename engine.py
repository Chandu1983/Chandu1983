import numpy as np

class SNAPBayesianEngine:
    def __init__(self, alpha=0.3, eps=1e-12):
        """
        ========================================================================
        गाडे फंडामेंटल बाउंड्री इंजन (FINAL PERFECT SHARP VERSION)
        ========================================================================
        लेखक: चंद्रकांत शिवराम गाडे (Chandrakant Shivram Gade)
        स्थान: नासिक, महाराष्ट्र, भारत (Nashik, India)
        ========================================================================
        """
        self.alpha = alpha
        self.eps = eps
        self.c = 299792458.0          
        self.G = 6.67430e-11          
        self.hbar = 1.0545718e-34     
        self.l_p_sq = (self.hbar * self.G) / (self.c**3)

    def flux(self, mag):
        mag = np.asarray(mag, dtype=float)
        return 10 ** (-0.4 * mag)

    def normalize(self, x):
        x = np.asarray(x, dtype=float)
        return x / (np.mean(x) + self.eps)

    def compute_state(self, mag):
        F = self.flux(mag)
        S = self.normalize(F)
        dS = np.gradient(S) if len(S) >= 2 else np.zeros_like(S)
        pi_B = S
        pi_T = 1.0 + self.alpha * dS
        return pi_B, pi_T

    def index(self, pi_B, pi_T):
        return np.asarray(pi_B, dtype=float) - np.asarray(pi_T, dtype=float)

    def probability(self, I):
        I = np.asarray(I, dtype=float)
        if len(I) < 2:
            return np.full_like(I, 0.5, dtype=float)
        dI = np.gradient(I)
        vol = np.array([np.std(I[max(0, i - 5):i + 1]) for i in range(len(I))])
        z = 1.2 * I + 0.8 * dI + 0.6 * vol
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, mag):
        pi_B, pi_T = self.compute_state(mag)
        I = self.index(pi_B, pi_T)
        P = self.probability(I)
        return pi_B, pi_T, I, P

def filter_gade_events(P, threshold, window=15):
    """
    स्मार्ट लोकल पीक फ़िल्टर (Non-Maximum Suppression)
    यह लगातार आने वाले क्लस्टर्स में से केवल सबसे ऊंचे असली पॉइंट को चुनता है।
    """
    P = np.asarray(P, dtype=float)
    raw_indices = np.where(P > threshold)
    filtered_indices = []
    
    for idx in raw_indices[0]:
        start = max(0, idx - window)
        end = min(len(P), idx + window + 1)
        if P[idx] == np.max(P[start:end]):
            if int(idx) not in filtered_indices:
                filtered_indices.append(int(idx))
                
    return filtered_indices
