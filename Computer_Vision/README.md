# FASTER-RCNN Implementation

[**WORK IN PROGRESS**]

## Description

### General Naming Conventions

- **Variables and Functions**: Use `snake_case`
- **Classes**: Use `PascalCase`
- **Constants**: Use `UPPERCASE_WITH_UNDERSCORES`
- Use `snake_case` for filenames
- Use descriptive names that convey the purpose

### System Design for the Project

- In this project we build a simple user facing WebApp, that will generate object detection predictions identifying Pedestrians.
- The app will be containerized in a Docker container
- Deployment can be done to any container registry such as AWS ECR or Docker Hub
- We can have an orchestration system such as Airflow that spins up the inference service, pulls the container from the registry and runs the inference application

### Class diagram

```
,-------------------------------------------------------.
|DataLoader                                             |
|-------------------------------------------------------|
|- data_path: Masks & Images Parent Dir                 |
|+ load_data(): Loads Object Detection Dataset from dir |
`-------------------------------------------------------'
                            |
                            |
       ,----------------------------------------.
       |FasterRCNNModel                         |
       |----------------------------------------|
       |- model: Any                            |
       |- checkpoints_path: str                 |
       |+ train_model(data: Dataset): None      |
       |+ load_model(path: str): None           |
       |+ save_model(path: str): None           |
       |+ predict(image: np.ndarray): List[Dict]|
       `----------------------------------------'
                            |
    ,----------------------------------------------.
    |InferenceService                              |
    |----------------------------------------------|
    |- model: FasterRCNNModel                      |
    |+ run_inference(image: np.ndarray): List[Dict]|
    `----------------------------------------------'
                            |

         ,-------------------------------------.
         |FlaskApp                             |
         |-------------------------------------|
         |- app: Flask                         |
         |- inference_service: InferenceService|
         |+ upload_image(): Response           |
         |+ get_prediction(): Response         |
         `-------------------------------------'
```

## Installation

## Usage

## Contributing

## License

## PLAN

- [x] Create a Dataset class, that initializes data splits
- [x] Create a BackBone Registry for the feature extracting cnn backbones

- [ ] Create a RPN having the following functionalities:
  - [ ] One function taking input of feature map, anchor bbox information (scales, ratios) -> Returning the Valid Anchor Bboxes
  - [ ] One function taking the Valid Anchor Bboxes, assigning positive and negative labels as per the paper based on IoUs. -> Returning Labels & Bboxes
  - [ ] One function generating Target Anchor Bboxes & Valid Labels to Train the RPN
  - [ ] One Function to Encode & Decode Translations of Bboxes
  - [ ] One Training RPN Function

- [ ] Creating a Detection Network
  - [ ] One Function Taking input from RPN, and performing ROI steps
  - [ ] One Function to do the bbox format conversions
  - [ ] One Function to do RoI pooling
  - [ ] One Function Generating Proposal Targets
  - [ ] One Function to Train the Detection Network
