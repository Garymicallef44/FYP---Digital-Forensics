import cv2
import numpy as np


def SIFTfeatures(image):
    # Create a sift object and compute keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def matchFeatures(desc1, desc2):
    # Using FLANN based matcher
    indexparams = dict(algorithm=1, trees=5)  # Using KDTree with 5 trees
    searchparams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexparams, searchparams)
    matches = flann.knnMatch(desc1, desc2, k=2)  # Find the 2 nearest neighbors for each descriptor
    return matches


def filterMatches(matches, ratio=0.75):
    good = []
    for m, n in matches:  # Use Lowe's ratio test, where m is the closest and n is the second closest match
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def similarityScore(goodmatches, keypoints1, keypoints2):
    # Compare the number of good matches to the number of availible keypoints. this creates the similarity score as a %.
    totalfeatures = min(len(keypoints1), len(keypoints2))
    if totalfeatures == 0:
        return 0
    return len(goodmatches) / totalfeatures * 100


# Load images
img1 = cv2.imread(
    r"C:\\Users\\User\\Documents\\GitHub\\FYP---Digital-Forensics\\Similar images to personal\\tritonfountain.jpg")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(r"C:\\Users\\User\\Documents\\GitHub\\FYP---Digital-Forensics\\Similar images to personal\\triton-1.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

print(img1)
print(img2)

# Compute keypoints
keypoints1, desc1 = SIFTfeatures(img1)
keypoints2, desc2 = SIFTfeatures(img2)

# Find matches and filter good matches
matches = matchFeatures(desc1, desc2)
good = filterMatches(matches)

# Calculate similarity score
score = similarityScore(good, keypoints1, keypoints2)
print(f"The similarity score is: {score}%")

# Visualise matches by drawing lines between corresponding keypoints
visualisation = cv2.drawMatches(img1, img2, keypoints1, keypoints2, good, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display result
cv2.imshow("Matches", visualisation)
cv2.waitKey(0)
cv2.destroyAllWindows()
