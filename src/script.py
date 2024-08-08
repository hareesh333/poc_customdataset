# import xml.etree.ElementTree as ET

# # Example data
# annotations = [
#     (6, 0.577692, 0.695901, 0.041087, 0.055291),
#     (7, 0.375571, 0.731633, 0.040347, 0.05646),
#     (7, 0.621637, 0.591445, 0.040999, 0.055764),
#     (6, 0.158622, 0.733167, 0.040564, 0.056357),
#     (6, 0.57722, 0.261291, 0.040711, 0.055639),
#     (6, 0.787178, 0.389176, 0.040803, 0.055627),
#     (6, 0.716994, 0.502155, 0.041045, 0.055211),
#     (6, 0.926733, 0.649162, 0.040531, 0.055724),
# ]

# # Image dimensions
# img_width = 800
# img_height = 600

# # Define the XML structure
# def create_xml(annotation_list, img_width, img_height, filename, filepath):
#     annotation = ET.Element("annotation")
    
#     folder = ET.SubElement(annotation, "folder")
#     folder.text = "pytorchdataset"
    
#     file_name = ET.SubElement(annotation, "filename")
#     file_name.text = filename
    
#     path = ET.SubElement(annotation, "path")
#     path.text = filepath
    
#     source = ET.SubElement(annotation, "source")
#     database = ET.SubElement(source, "database")
#     database.text = "Unknown"
    
#     size = ET.SubElement(annotation, "size")
#     width = ET.SubElement(size, "width")
#     width.text = str(img_width)
#     height = ET.SubElement(size, "height")
#     height.text = str(img_height)
#     depth = ET.SubElement(size, "depth")
#     depth.text = "3"
    
#     segmented = ET.SubElement(annotation, "segmented")
#     segmented.text = "0"
    
#     for obj in annotation_list:
#         class_id, x_center, y_center, width, height = obj
        
#         # Convert normalized coordinates to pixel coordinates
#         xmin = int((x_center - width / 2) * img_width)
#         ymin = int((y_center - height / 2) * img_height)
#         xmax = int((x_center + width / 2) * img_width)
#         ymax = int((y_center + height / 2) * img_height)
        
#         obj_element = ET.SubElement(annotation, "object")
        
#         name = ET.SubElement(obj_element, "name")
#         name.text = f"Class_{class_id}"  # Replace with your class names if available
        
#         pose = ET.SubElement(obj_element, "pose")
#         pose.text = "Unspecified"
        
#         truncated = ET.SubElement(obj_element, "truncated")
#         truncated.text = "0"
        
#         difficult = ET.SubElement(obj_element, "difficult")
#         difficult.text = "0"
        
#         bndbox = ET.SubElement(obj_element, "bndbox")
#         xmin_elem = ET.SubElement(bndbox, "xmin")
#         xmin_elem.text = str(xmin)
#         ymin_elem = ET.SubElement(bndbox, "ymin")
#         ymin_elem.text = str(ymin)
#         xmax_elem = ET.SubElement(bndbox, "xmax")
#         xmax_elem.text = str(xmax)
#         ymax_elem = ET.SubElement(bndbox, "ymax")
#         ymax_elem.text = str(ymax)
    
#     return annotation

# # Create XML
# filename = "page_390.jpg"
# filepath = "/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/Microcontroller Detection/train/page_390.jpg"
# xml_data = create_xml(annotations, img_width, img_height, filename, filepath)

# # Convert to string and print
# xml_str = ET.tostring(xml_data, encoding='unicode')
# print(xml_str)

# # Optional: Save to file
# with open("annotation.xml", "w") as file:
#     file.write(xml_str)
import os
import xml.etree.ElementTree as ET

# Define the folder containing TXT files
txt_folder = '/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/Microcontroller Detection/test'
xml_folder = '/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/Microcontroller Detection/train/xml_files'
img_width = 800
img_height = 600

# Ensure XML folder exists
os.makedirs(xml_folder, exist_ok=True)

def convert_txt_to_xml(txt_file_path, img_width, img_height, filename, filepath):
    annotation_list = []
    
    with open(txt_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                annotation_list.append((class_id, x_center, y_center, width, height))
    
    # Create XML structure
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder")
    folder.text = "pytorchdataset"
    
    file_name = ET.SubElement(annotation, "filename")
    file_name.text = filename
    
    path = ET.SubElement(annotation, "path")
    path.text = filepath
    
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_width)
    height = ET.SubElement(size, "height")
    height.text = str(img_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    for obj in annotation_list:
        class_id, x_center, y_center, width, height = obj
        
        # Convert normalized coordinates to pixel coordinates
        xmin = int((x_center - width / 2) * img_width)
        ymin = int((y_center - height / 2) * img_height)
        xmax = int((x_center + width / 2) * img_width)
        ymax = int((y_center + height / 2) * img_height)
        
        obj_element = ET.SubElement(annotation, "object")
        
        name = ET.SubElement(obj_element, "name")
        name.text = f"Class_{int(class_id)}"  # Replace with your class names if available
        
        pose = ET.SubElement(obj_element, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj_element, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj_element, "difficult")
        difficult.text = "0"
        
        bndbox = ET.SubElement(obj_element, "bndbox")
        xmin_elem = ET.SubElement(bndbox, "xmin")
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, "ymin")
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, "xmax")
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, "ymax")
        ymax_elem.text = str(ymax)
    
    return annotation

def process_all_txt_files(txt_folder, xml_folder, img_width, img_height):
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            txt_file_path = os.path.join(txt_folder, txt_file)
            filename = txt_file.replace('.txt', '.jpg')  # Assuming image file has the same name but different extension
            filepath = os.path.join(txt_folder, filename)
            
            xml_data = convert_txt_to_xml(txt_file_path, img_width, img_height, filename, filepath)
            
            xml_str = ET.tostring(xml_data, encoding='unicode')
            xml_file_path = os.path.join(xml_folder, filename.replace('.jpg', '.xml'))
            
            with open(xml_file_path, 'w') as file:
                file.write(xml_str)

# Process all TXT files
process_all_txt_files(txt_folder, xml_folder, img_width, img_height)
print("converted successfull...")
