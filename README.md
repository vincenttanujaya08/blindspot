# ADAS Blind Spot Detection System

## Overview
This project implements a **vision-based Advanced Driver Assistance System (ADAS)** for real-time blind spot detection using multiple cameras.

The system monitors the environment around a vehicle and detects potential hazards such as nearby vehicles approaching or entering the blind spot.

The system processes video streams from four cameras:

- Front camera
- Rear camera
- Left side camera
- Right side camera

Using **real-time object detection and motion analysis**, the system estimates the proximity and movement of detected objects and determines whether they pose a risk to the vehicle.

---

## System Architecture

The system follows a computer vision pipeline:

```
Camera Input
     ↓
Frame Processing
     ↓
Object Detection (YOLO)
     ↓
Bounding Box Extraction
     ↓
Motion & Position Analysis
     ↓
Proximity Scoring
     ↓
Threat Classification
     ↓
Driver Warning
```

The goal of this pipeline is not only to detect objects but also to **understand spatial context and potential collision risk**.

---

## Object Detection

Object detection is performed using the **YOLO (You Only Look Once)** model through the Ultralytics library.

For each frame, the model returns:

- object class (car, motorcycle, person, etc.)
- confidence score
- bounding box coordinates `(x1, y1, x2, y2)`

From the bounding box we compute:

```
width  = x2 - x1
height = y2 - y1
area   = width * height
```

The bounding box represents the object's location and size in the frame.

YOLO was chosen because it provides:

- real-time detection speed
- single-pass inference
- low latency suitable for ADAS systems

---

## Distance Approximation Using Bounding Box Size

In monocular vision systems, actual depth information is not available. Therefore the system uses **bounding box size as an approximation of object distance**.

Perspective principle:

```
object_size ∝ 1 / distance
```

Meaning:

- larger bounding box → object closer to the camera  
- smaller bounding box → object farther away  

Example:

```
area(t)   = 1200
area(t+1) = 2000
```

If the area increases significantly, the object is likely approaching the vehicle.

---

## Position-Based Proximity Estimation

Object position within the frame also provides distance information.

For vehicle-mounted cameras:

- objects closer to the vehicle appear lower in the frame  
- objects farther away appear near the top of the frame  

The system computes the vertical center of the object:

```
center_y = (y1 + y2) / 2
```

Interpretation:

- higher `center_y` → object closer  
- lower `center_y` → object farther  

---

## Motion Analysis

To determine whether an object is moving toward the vehicle, the system performs motion analysis.

For lateral motion detection the system uses **Farneback Optical Flow**.

Optical flow estimates pixel movement between frames.

If a pixel moves from:

```
(x, y) → (x + dx, y + dy)
```

its motion vector is:

```
v = (dx, dy)
```

By analyzing these vectors, the system determines whether objects move:

- toward the vehicle
- away from the vehicle
- parallel to the vehicle

---

## Camera-Specific Detection Logic

Different cameras use different proximity estimation strategies because the **type of threat varies by direction**.

### Front Camera
Focus: collision risk with objects ahead.

Detection logic:

- vertical position in frame
- bounding box size
- bounding box growth over time

Objects appearing lower in the frame and increasing in size indicate possible collision risk.

### Rear Camera
Focus: approaching vehicles from behind.

Detection logic:

- bounding box size
- bounding box growth
- vertical proximity

### Left Camera
Focus: vehicles approaching the blind spot.

Detection logic:

- bounding box growth across frames
- object proximity to vehicle side

### Right Camera
Focus: lateral motion into the vehicle lane.

Detection logic:

- optical flow motion vectors
- direction of object movement

If motion vectors indicate the object moves toward the vehicle path, the risk level increases.

---

## Proximity Scoring

The system combines multiple indicators into a **proximity score**.

```
score =
w1 * size_factor +
w2 * motion_factor +
w3 * position_factor
```

Example weights:

```
w1 = 0.4
w2 = 0.4
w3 = 0.2
```

Where:

- **size_factor** represents bounding box size
- **motion_factor** represents movement toward the vehicle
- **position_factor** represents object location in the frame

---

## Threat Classification

Based on the computed score, the system classifies the risk level:

```
SAFE      : no object in danger zone
WARNING   : object detected near vehicle
DANGER    : object very close or approaching vehicle
```

These classifications are used to trigger warnings for the driver.

---

## Real-Time Optimization

### Frame Skipping
Instead of processing every frame:

```
30 FPS input
↓
10 FPS processed
```

This reduces computational load while maintaining responsiveness.

### Exponential Moving Average (EMA)

Used to smooth motion metrics:

```
EMA = α * current + (1 − α) * previous
```

This prevents sudden fluctuations in system indicators.

---

## Limitations

As a vision-only system:

- cannot measure exact physical distance
- performance depends on lighting conditions
- accuracy depends on detection quality

---

## Future Improvements

Possible improvements include:

- object tracking (SORT / DeepSORT)
- trajectory prediction
- camera calibration for better distance estimation
- multi-sensor fusion (radar or lidar)
