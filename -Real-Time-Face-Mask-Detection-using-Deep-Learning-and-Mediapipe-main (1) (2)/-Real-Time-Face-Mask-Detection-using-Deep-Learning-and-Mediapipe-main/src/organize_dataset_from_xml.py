import os
import shutil
import xml.etree.ElementTree as ET

# ===============================
# CONFIGURATION
# ===============================
images_path = "../dataset/images"
annotations_path = "../dataset/annotations"
output_path = "../dataset_sorted"

# Create output directories
os.makedirs(os.path.join(output_path, "with_mask"), exist_ok=True)
os.makedirs(os.path.join(output_path, "without_mask"), exist_ok=True)
os.makedirs(os.path.join(output_path, "mask_weared_incorrect"), exist_ok=True)

# ===============================
# READ ALL XML FILES
# ===============================
for xml_file in os.listdir(annotations_path):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(annotations_path, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get corresponding image filename
    image_name = root.find("filename").text
    image_path = os.path.join(images_path, image_name)

    # Extract label (mask category)
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()

        if label == "with_mask":
            dest_folder = "with_mask"
        elif label == "without_mask":
            dest_folder = "without_mask"
        elif label == "mask_weared_incorrect":
            dest_folder = "mask_weared_incorrect"
        else:
            continue

        dest_path = os.path.join(output_path, dest_folder)
        os.makedirs(dest_path, exist_ok=True)

        if os.path.exists(image_path):
            shutil.copy(image_path, dest_path)
        break  # one label per image

print("\n✅ Dataset successfully organized!")
print(f"Saved sorted dataset in: {output_path}")
