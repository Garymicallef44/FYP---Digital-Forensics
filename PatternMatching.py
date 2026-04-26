import cv2
import numpy as np
import os
from ProvenanceEvidence import ProvenanceEvidence


def loadImage(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: File does not exist in: {path}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Error: Could not read image: {path}")

    pixelcount = img.shape[0] * img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    enhanced = clahe.apply(gray)

    max_dim = 1024
    h, w = enhanced.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        enhanced = cv2.resize(enhanced, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)

    return enhanced, pixelcount


def extractSIFT(image):
    sift = cv2.SIFT_create(nfeatures = 3000, contrastThreshold = 0.03, edgeThreshold = 10)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def matchFeatures(desc1, desc2):
    if desc1 is None or desc2 is None:
        return [], []

    kdtree = 1
    indexparams = dict(algorithm = kdtree, trees = 5)
    searchparams = dict(checks = 100)
    flann = cv2.FlannBasedMatcher(indexparams, searchparams)

    forward = flann.knnMatch(desc1, desc2, k = 2)
    backward = flann.knnMatch(desc2, desc1, k = 2)
    return forward, backward

def geometricVerification(kp1, kp2, goodMatches, reprojthresh = 3.0):
    if len(goodMatches) < 4:
        return [], None

    srcpts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    dstpts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(srcpts, dstpts, cv2.RANSAC, reprojthresh)

    if mask is None:
        return [], None

    inliers = [goodMatches[i] for i in range(len(goodMatches)) if mask[i]]
    return inliers, H


def validateHomography(H):
    if H is None:
        return 0.0

    det = np.linalg.det(H)
    if det < 0.1 or det > 10.0:
        return 0.0

    svdvals = np.linalg.svd(H, compute_uv = False)
    cond = svdvals[0] / (svdvals[-1] + 0.0000001)
    if cond > 1000:
        return 0.0

    detquality = 1.0 / (1.0 + abs(np.log(det + 0.0000001)))
    condquality = 1.0 / (1.0 + np.log1p(max(0.0, cond - 1.0) / 100.0))
    return detquality * condquality


def similarityScore(inliers, goodMatches, H = None):
    if not goodMatches:
        return 0.0, 0.0

    n_inliers = len(inliers)
    n_good = len(goodMatches)

    inlierratio = n_inliers / n_good

    countconfidence = 1.0 - np.exp(-n_inliers / 20.0)

    if inliers:
        avgdist = np.mean([m.distance for m in inliers])
        distquality = max(0.0, 1.0 - avgdist / 300.0)
    else:
        distquality = 0.0

    hquality = validateHomography(H)

    score = (0.30 * inlierratio + 0.30 * countconfidence + 0.20 * distquality + 0.20 * hquality) * 100

    return score, hquality


def ratioFilter(matches, ratio = 0.75):
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def coreMatch(img1, kp1, desc1, img2, kp2, desc2):
    forward, _ = matchFeatures(desc1, desc2)
    goodMatches = ratioFilter(forward)
    inliers, H = geometricVerification(kp1, kp2, goodMatches)
    score, hquality = similarityScore(inliers, goodMatches, H)

    img2_inv = cv2.bitwise_not(img2)
    kp2_inv, desc2_inv = extractSIFT(img2_inv)
    fwd_inv, _ = matchFeatures(desc1, desc2_inv)
    goodMatches_inv = ratioFilter(fwd_inv)
    inliers_inv, H_inv = geometricVerification(kp1, kp2_inv, goodMatches_inv)
    score_inv, hquality_inv = similarityScore(inliers_inv, goodMatches_inv, H_inv)

    if score_inv > score:
        kp2, goodMatches, inliers, H, score, hquality = kp2_inv, goodMatches_inv, inliers_inv, H_inv, score_inv, hquality_inv
        img2 = img2_inv

    det = float(np.linalg.det(H)) if H is not None else 0.0
    return kp2, goodMatches, inliers, H, score, hquality, det, img2


def compareImages(path1, path2, showMatches = False):
    print(f"\n Comparing: \n - {path1}\n - {path2}")

    img1, pixels1 = loadImage(path1)
    img2, pixels2 = loadImage(path2)
    kp1, desc1 = extractSIFT(img1)
    kp2, desc2 = extractSIFT(img2)

    print(f"Keypoints: {len(kp1)} vs {len(kp2)}")

    kp2, goodMatches, inliers, H, score, hquality, det, img2 = coreMatch(
        img1, kp1, desc1, img2, kp2, desc2)

    print(f"Similarity Score: {score:.2f}%")
    print(f"Good matches (forward ratio): {len(goodMatches)}")
    print(f"Inliers (after RANSAC): {len(inliers)}")
    if goodMatches:
        print(f"Inlier ratio: {len(inliers)/len(goodMatches):.2%}")
    print(f"Homography quality: {hquality:.4f}")

    if showMatches == True:
        width = 4000
        height = 1800
        vis = cv2.drawMatches(img1, kp1, img2, kp2, inliers, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        h,w = vis.shape[:2]
        scale = min(width / w, height / h)
        resized_vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.namedWindow("Feature matches", cv2.WINDOW_NORMAL)
        cv2.imshow("Feature matches", resized_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ProvenanceEvidence(score, len(inliers), len(goodMatches), len(kp1), len(kp2), det, pixels1, pixels2)


def compareFromFeatures(img1, kp1, desc1, pixels1, img2, kp2, desc2, pixels2):
    kp2, goodMatches, inliers, H, score, hquality, det, _ = coreMatch(img1, kp1, desc1, img2, kp2, desc2)

    return ProvenanceEvidence(score, len(inliers), len(goodMatches), len(kp1), len(kp2), det, pixels1, pixels2)


if __name__ == "__main__":
    imgA = r"C:\Users\User\Documents\GitHub\FYP---Digital-Forensics\TestPath\ringroad.jpg" # Original img
    imgB = r"C:\Users\User\Documents\GitHub\FYP---Digital-Forensics\TestPath\ringroad2.jpg" # Derivation img

    evidence = compareImages(imgA, imgB, showMatches = True)
    print(f"Similarity score: {evidence.score:.2f}  \n Inliers: {evidence.inliers}  \nGood matches: {evidence.good_matches}")