import cv2
import numpy as np
import matplotlib.pyplot as plt
def display_image(title ,image):
    plt.figure(figsize = (8, 8))
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
def interactive_edge_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image("Grayscale Image", gray_image)
    print("select an option:")
    print("1.Sobel Edge Detection")
    print("2.Canny Edge Detection")
    print("3.Laplacian Edge Detection")
    print("4.Gaussian smoothing")
    print("5.Median Fliltering")
    print("6.Exit")
    while True:
        choice = input("Enter your choice (1-6): ")
        if choice == '1':
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            combined_sobel = cv2.bitwise_or(sobel_x.astype(np.uint8), sobel_y.astype(np.uint8))
            display_image("Sobel Edge Detection", combined_sobel)
        elif choice == '2':
            print("Enter the lower and upper thresholds for Canny Edge Detection:")
            lower_threshold = int(input("Lower threshold: "))
            upper_threshold = int(input("Upper threshold: "))
            canny_edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
            display_image("Canny Edge Detection", canny_edges)
        elif choice == '3':
            laplacian_edges = cv2.Laplacian(gray_image, cv2.CV_64F)
            display_image("Laplacian Edge Detection", np.abs(laplacian_edges.astype(np.uint8)))
        elif choice == '4':
            print("Enter the kernel size for Gaussian smoothing (must be odd):")
            kernel_size = int(input("Kernel size: "))
            blurred = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
            display_image("Gaussian Smoothing", blurred)

