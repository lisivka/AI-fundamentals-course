import cv2
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
from urllib.request import urlopen
from io import BytesIO
from PIL import Image

def detect_dominant_color(image_path):
    # Read the image
    response = urlopen(image_path)
    image_data = BytesIO(response.read())
    img = Image.open(image_data)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img = io.imread(image_path)
    # cv2_imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


    # Flatten the image

    pixels = img.reshape((-1, 3))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10, max_iter=100)
    kmeans.fit(pixels)

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_.astype(int)[0]

    # Convert BGR to RGB
    dominant_color = (dominant_color[2], dominant_color[1], dominant_color[0])

    # Calculate the percentage of dominant color's presence
    percentage = (np.sum(kmeans.labels_ == kmeans.labels_[0]) / len(kmeans.labels_)) * 100

    return dominant_color, percentage

# Path to the input image
image_path = "https://raw.githubusercontent.com/mehalyna/cooltest/master/cooltest/sunflowers.png"

# Call the function and get the results
dominant_color, percentage = detect_dominant_color(image_path)

# Print the results
print(f"Dominant Color: {dominant_color}")
print(f"Percentage: {percentage:.2f}%")
