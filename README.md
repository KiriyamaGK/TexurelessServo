# AlignAnything
## **Setup Python Path**
```
export PYTHONPATH="/path/to/your/modules:$PYTHONPATH"
```

## **Training** 

```
python experiments/experiment.py
```

## **Rollout** 
### Simulation Environment
```
python experiments/rollout.py
```

### Real-world Environment
```
python experiments/rollout_real.py
```

## **Data Collection** 
```
python data/collect_demo.py
```

## **Real-world Data Augmentation** 
### 1. Prepare YOLO Training Data
```
python data/make_yolo_raw_dataset.py
```

### 2. Data Annotation with RoboFlow
1. Upload the generated YOLO format data to RoboFlow platform
2. Perform data annotation and augmentation
3. Export the annotated dataset

### 3. Train YOLO Model
```
python train_yolo.py
```

### 4. Create Augmented HDF5 Dataset
```
python data/make_augmented_hdf5.py
```

## Collision Detection Module Installation

This module is implemented in C++ with Python bindings via pybind11. It requires compilation.

### Prerequisites
- CMake (>= 3.10)
- Make
- Python development headers (python3-dev)
- pybind11 (will be installed via pip)

### Installation Steps
1. Install Python dependencies:
   ```bash
   pip install numpy trimesh pybind11
   ```

2. Compile the C++ module:
   ```bash
   cd real/collision_detection
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Verify installation:
   ```bash
   cd ..
   python -c "import sdf_module; print(sdf_module.__doc__)"
   ```
   This should print the help message of the module.

### Notes
- The compilation must be done in the `collision_detection` directory as described
- If you encounter issues with pybind11 not being found, check the installation path of pybind11 (using `pip show pybind11`) and ensure it is in CMake's search path
- When using the module in Python, make sure to add `real/collision_detection` to your PYTHONPATH


