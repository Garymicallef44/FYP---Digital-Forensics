from dataclasses import dataclass

@dataclass
class ProvenanceEvidence:
    score: float
    inliers: int
    good_matches: int
    kp1: int
    kp2: int
    homography_det: float
    img1pixels: int = 0
    img2pixels: int = 0