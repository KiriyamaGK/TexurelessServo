# AlignAnything

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
python data/collect_demo_offline.py
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


