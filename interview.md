1. How did you train EfficientNet-B4?
Ans: I used transfer learning with ImageNet pretrained weights. Initially, I froze all EfficientNet-B4 feature extraction layers and trained only the custom classification head for the mask/no-mask task. After the classifier converged, I fine-tuned the upper EfficientNet blocks by unfreezing the last layers and training them with a smaller learning rate. This allowed the model to adapt high-level ImageNet features to face mask detection while reducing overfitting and training time.

2. challenges which i face from starting to ending
Ans: 

3. how many time i train and model improve step by step
Ans: 





---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

# Dataclass Interview Questions

## Q1. What is a dataclass in Python?

### Answer
A dataclass is a special decorator that automatically generates methods like:
- __init__()
- __repr__()
- __eq__()

Example:

```python
from dataclasses import dataclass

@dataclass
class Student:
    name: str
    age: int
```

---

## Q2. Why use dataclass instead of a normal class?
### Answer
It reduces boilerplate code and makes classes cleaner.

## Q3. What is field() in dataclass?
### Answer
field() is used to customize dataclass attributes.
Example:
```python
from dataclasses import field

marks: list = field(default_factory=list)
```

---
## Q4. What is default_factory?
### Answer
Used to create a new mutable object for each instance.

```python
marks: list = field(default_factory=list)
```
Without default_factory, all objects may share the same list.
---

## Q5. What is __post_init__()?
### Answer
A method that runs automatically after the generated __init__().

# Q. What does your Data Ingestion module do?
## Answer
1. Reads dataset paths from configuration.
2. Creates artifact directories.
3. Copies train, validation, and test datasets.
4. Logs all operations.
5. Handles exceptions using a custom exception class.
6. Returns dataset paths through a DataIngestionArtifact object.

# Q. Why use shutil.copytree() in Data Ingestion?
## Answer
`shutil.copytree()` is used to copy an entire directory along with all subdirectories and files from the source location to the artifact location.
It helps preserve the original dataset while creating a separate working copy for the ML pipeline.
---

# Q. Why not use os.makedirs() instead of shutil.copytree()?
## Answer
`os.makedirs()` only creates empty directories.
It does not copy files or subfolders.
`shutil.copytree()` creates the directory and copies all files and subdirectories, making it suitable for dataset ingestion.