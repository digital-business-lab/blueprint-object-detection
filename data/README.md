# Custom Dataset
If you have a custom dataset, the structure has to be the same as in the roboflow datasets. 

## Inside folder ./data:
- Create your dataset folder (Look at Dataset structure)

## Inside config.yaml change:
- Set Roboflow to 1
- Set project_name to to "your_folder_name"
- Set PredictionPath
- Set PredictionSavePath
- Set ImageType

## Inside template.yaml
- Change variables
- Rename file to "data.yaml"

## Dataset structure
![Folder Structure](../pictures/folder_structure.png)
