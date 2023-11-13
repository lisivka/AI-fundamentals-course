import cv2
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# @test_detect_color_task
def detect_dominant_color(image_path):
    # Read the image
    response = urlopen(image_path)
    image_data = BytesIO(response.read())
    img = Image.open(image_data)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Поменяйте цветовое пространство на BGR2RGB, если оригинальное изображение в цветовом пространстве BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # show the image
    plt.imshow(img)

    # Flatten the image
    pixels = img.reshape((-1, 3))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=SIZE, random_state=0,n_init=10, max_iter=200)
    kmeans.fit(pixels)

    dominant_color = ()
    dominant_percentage = 0
    # Visualize dominant colors from each cluster
    plt.figure(figsize=(10, 3))
    for i in range(SIZE):
        cluster_color = kmeans.cluster_centers_.astype(int)[i]
        cluster_color_rgb = (cluster_color[2], cluster_color[1], cluster_color[0])

        # Calculate the percentage of colors in each cluster
        percentage = (np.sum(kmeans.labels_ == i) / len(kmeans.labels_)) * 100

        #  Get the dominant color and its percentage
        if dominant_percentage < percentage:
            dominant_percentage = percentage
            dominant_color = cluster_color_rgb

        print(f'Cluster {i+1} color: {cluster_color_rgb}    Percentage: {percentage:.2f}%'  )

        plt.subplot(1, SIZE, i + 1)
        plt.imshow([[cluster_color_rgb]])
        plt.axis('off')
        plt.title(f'#{i+1} color {cluster_color_rgb}  \n Percentage:'
                  f' {percentage:.2f}%', fontdict={'fontsize': 10})

    plt.show()

    return dominant_color, dominant_percentage

SIZE = 3
# Path to the input image
image_path = "https://raw.githubusercontent.com/mehalyna/cooltest/master/cooltest/sunflowers.png"

# Call the function and get the results
dominant_color, percentage = detect_dominant_color(image_path)

# Print the results
print(f"Dominant Color: {dominant_color}")
print(f"Percentage: {percentage:.2f}%")
