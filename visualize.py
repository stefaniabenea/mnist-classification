import matplotlib.pyplot as plt

def plot_wrong_predictions(wrong_images,wrong_labels, wrong_predictions, num_to_show=8):
    plt.figure(figsize=(12,4))
    for i in range(num_to_show):
        plt.subplot(2,4,i+1)
        plt.imshow(wrong_images[i].squeeze(),cmap='gray')
        plt.title(f"Correct label: {wrong_labels[i].item()}\n Prediction: {wrong_predictions[i].item()}")
    plt.tight_layout()
    plt.show()

def plot_sample_images(images, labels, predictions, num_to_show=8):
    plt.figure(figsize=(12,4))
    for i in range(num_to_show):
        plt.subplot(2,4,i+1)
        plt.imshow(images[i].squeeze(),cmap='gray')
        plt.title(f"Correct label: {labels[i].item()}\n Predicted label: {predictions[i].item()}")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
