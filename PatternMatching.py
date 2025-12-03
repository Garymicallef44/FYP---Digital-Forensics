import cv2
import os


def loadimage(path):
    # Load an image and convert to grayscale, with error handling for debugging
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: File does not exist in: {path}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Error: Could not read image: {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extractSIFT(image):
    # Feature extraction
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)

    return keypoints,descriptors

def matchfeatures(desc1, desc2):
    # Uses KD-tree for fast feature matching
    indexParam = dict(algorithm = 1, trees = 5)
    searchParam = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(indexParam,searchParam)
    return flann.knnMatch(desc1,desc2,k=2)

def filtermatches(matches, ratio = 0.75):
    # Rejects accidental matches, bg noise and repetitive texture false matches
    good = []

    for m,n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def similarityscore(goodMatches, kp1, kp2):
    # Measures score relative to the one with less keypoints
    total = min(len(kp1), len(kp2))

    if total == 0:
        return 0
    return (len(goodMatches) / total) * 100


def compareimages(path1, path2, showMatches = True):
    # Load the images, extract features, use FLANN matching, filter matches, compute similarity percentage and show results
    print(f"\n Comparing: \n - {path1}\n - {path2}")

    img1 = loadimage(path1)
    img2 = loadimage(path2)
    kp1, desc1 = extractSIFT(img1)
    kp2, desc2 = extractSIFT(img2)

    print(f"Keypoints: {len(kp1)} vs {len(kp2)}")

    rawMatches = matchfeatures(desc1, desc2)
    goodMatches = filtermatches(rawMatches)

    score = similarityscore(goodMatches, kp1, kp2)

    print(f"Similarity Score: {score: 2f}%")
    print(f"Good matches: {len(goodMatches)}")

    if showMatches:
        vis = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Feature matches", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return score

imgA = r"C:\Users\User\Documents\GitHub\FYP---Digital-Forensics\Personal dataset\unibutterfly.jpg"
imgB = r"C:\Users\User\Documents\GitHub\FYP---Digital-Forensics\Similar images to personal\61815597.jpg"

compareimages(imgA, imgB)