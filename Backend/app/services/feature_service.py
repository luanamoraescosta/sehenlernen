import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.transform import resize
import cv2
import pandas as pd
from skimage.feature import hog, corner_fast
from skimage.feature.texture import graycomatrix, graycoprops
from typing import Any, Optional
from fastapi import UploadFile
import io
from io import BytesIO




from app.services.data_service import load_image, get_all_image_ids, metadata_df, image_id_col


def generate_histogram_service(hist_type: str, image_index: int, all_images: bool) -> list[str]:
    """
    Generate histograms as base64 strings using OpenCV for accurate calculation.
    """
    img_ids = get_all_image_ids() if all_images else get_all_image_ids()[image_index:image_index+1]
    b64_list = []

    for img_id in img_ids:
        img_bytes = load_image(img_id)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue  # Skip invalid images

        plt.figure(figsize=(6, 4))
        if hist_type == "Black and White":
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.hist(gray_img.ravel(), 256, [0, 256], color='black', alpha=0.7)
            plt.title("Grayscale Histogram")
        else:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=color, alpha=0.7)
            plt.xlim([0, 256])
            plt.title("Color Histogram")
            plt.legend(["Blue", "Green", "Red"])

        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        b64_list.append(b64)

    return b64_list


def perform_kmeans_service(n_clusters: int, random_state: int, selected_images: list[int], use_all_images: bool) -> tuple[str, list[int]]:
    """
    Performs K-means clustering on the selected images and returns the plot and label assignments.
    """
    img_ids = get_all_image_ids()
    
    if not img_ids:
        raise Exception("No images available for clustering")
    
    # Determine which images to use
    if use_all_images or not selected_images:
        selected_ids = img_ids
        selected_indices = list(range(len(img_ids)))
    else:
        # Filter valid indices
        selected_indices = [i for i in selected_images if i < len(img_ids)]
        selected_ids = [img_ids[i] for i in selected_indices]
    
    if not selected_ids:
        raise Exception("No valid images selected for clustering")
    
    # Extract features from selected images
    features = []
    for img_id in selected_ids:
        img_bytes = load_image(img_id)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        r, g, b = img.split()
        hist_r = np.histogram(np.array(r), bins=32)[0]
        hist_g = np.histogram(np.array(g), bins=32)[0]
        hist_b = np.histogram(np.array(b), bins=32)[0]
        features.append(np.concatenate([hist_r, hist_g, hist_b]))
    
    # K-means processing
    X = np.vstack(features)
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(reduced)
    
    # Generate plot
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        ax.scatter(reduced[labels == i, 0], reduced[labels == i, 1], label=f"Cluster {i}", alpha=0.7)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='red')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    plot_b64 = base64.b64encode(buf.getvalue()).decode()
    
    return plot_b64, labels.tolist()


def extract_shape_service(method: str, image_index: int) -> tuple[list[Any], Optional[str]]:
    """
    Extract shape features and optional visualization.
    """
    img_ids = get_all_image_ids()
    img_bytes = load_image(img_ids[image_index])
    img = Image.open(io.BytesIO(img_bytes))
    features = []
    viz_b64 = None

    if method == "HOG":
        img_gray = img.convert('L')
        img_resized = resize(np.array(img_gray), (128, 64))
        fd, hog_image = hog(img_resized, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)
        features = fd.tolist()
        # Plot hog image
        fig, ax = plt.subplots()
        ax.imshow(hog_image, cmap='gray')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        viz_b64 = base64.b64encode(buf.getvalue()).decode()

    elif method == "SIFT":
        img_gray = img.convert('L')
        arr = np.array(resize(img_gray, (256,256))) * 255
        arr = arr.astype('uint8')
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(arr, None)
        features = des.tolist() if des is not None else []
        img_kp = cv2.drawKeypoints(arr, kp, None, color=(0,255,0))
        pil_kp = Image.fromarray(img_kp)
        buf = io.BytesIO()
        pil_kp.save(buf, format='PNG')
        viz_b64 = base64.b64encode(buf.getvalue()).decode()

    elif method == "FAST":
        img_gray = img.convert('L')
        arr = np.array(img_gray)
        kp = corner_fast(arr, threshold=30, nonmax_suppression=True)
        features = [list(point) for point in kp]
        img_kp = cv2.drawKeypoints(arr, [cv2.KeyPoint(x=float(p[1]), y=float(p[0]), _size=1) for p in kp], None, color=(0,255,0))
        pil_kp = Image.fromarray(img_kp)
        buf = io.BytesIO()
        pil_kp.save(buf, format='PNG')
        viz_b64 = base64.b64encode(buf.getvalue()).decode()

    return features, viz_b64


async def extract_haralick_service(train_images: list, train_labels: UploadFile, test_images: list) -> tuple[list, list]:
    """
    Extract Haralick features, train classifier, and predict test labels.
    """
    # Load training images
    X_train = []
    for f in train_images:
        data = await f.read()
        img = Image.open(io.BytesIO(data)).convert('L')
        arr = np.array(img)
        gm = graycomatrix(arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        props = graycoprops(gm, 'contrast')[0,0]
        X_train.append([props])
    # Load labels
    labels_df = pd.read_csv(train_labels.file)
    y_train = labels_df.iloc[:,0].tolist()
    # Train classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    # Extract test features
    X_test = []
    for f in test_images:
        data = await f.read()
        img = Image.open(io.BytesIO(data)).convert('L')
        arr = np.array(img)
        gm = graycomatrix(arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        props = graycoprops(gm, 'contrast')[0,0]
        X_test.append([props])
    preds = clf.predict(X_test).tolist()
    return y_train, preds


def extract_cooccurrence_service(image_index: int) -> list[float]:
    """
    Extract co-occurrence texture features for a given image.
    """
    img_ids = get_all_image_ids()
    img_bytes = load_image(img_ids[image_index])
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    arr = np.array(img)
    gm = graycomatrix(arr, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation']:
        feat = graycoprops(gm, prop).mean()
        features.append(float(feat))
    return features
