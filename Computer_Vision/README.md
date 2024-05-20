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
