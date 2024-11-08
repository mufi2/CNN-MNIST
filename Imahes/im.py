import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Define the path where images will be saved
save_path = "MNIST_Sample_Images"

# Create a transformation to convert images to tensor (for loading dataset)
transform = transforms.ToTensor()

# Load the MNIST dataset
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

# Ensure the main save path exists
os.makedirs(save_path, exist_ok=True)

# Dictionary to count saved images for each label
label_counts = {i: 0 for i in range(10)}
max_images_per_label = 10

# Loop through the dataset and save images
for idx, (image, label) in enumerate(train_loader):
    label = label.item()  # Get the label as an integer

    # Check if we already saved 10 images for this label
    if label_counts[label] < max_images_per_label:
        # Convert the image tensor to a PIL image
        pil_image = transforms.ToPILImage()(image.squeeze(0))

        # Create a folder for each label if it doesn't exist
        label_folder = os.path.join(save_path, f"label_{label}")
        os.makedirs(label_folder, exist_ok=True)

        # Save the image as PNG in the appropriate folder
        image_path = os.path.join(label_folder, f"{label}_{label_counts[label]}.png")
        pil_image.save(image_path)

        # Update the count for the current label
        label_counts[label] += 1

    # Stop if we have saved 10 images for each label
    if all(count >= max_images_per_label for count in label_counts.values()):
        break

print("Images saved successfully in the 'MNIST_Sample_Images' folder.")
