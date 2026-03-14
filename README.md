ADAS Blind Spot Detection System
Overview

This project implements a vision-based Advanced Driver Assistance System (ADAS) for real-time blind spot detection using multiple cameras.
The system monitors the environment around a vehicle and detects potential hazards such as nearby vehicles approaching or entering the blind spot.

The system processes video streams from four cameras:

Front camera

Rear camera

Left side camera

Right side camera

Using real-time object detection and motion analysis, the system estimates the proximity and movement of detected objects and determines whether they pose a risk to the vehicle.

System Architecture

The system follows a computer vision pipeline:

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

The goal of this pipeline is not only to detect objects but also to understand spatial context and potential collision risk.

Object Detection

Object detection is performed using the YOLO through the Ultralytics library.

For each frame, the model returns:

object class (car, motorcycle, person, etc.)

confidence score

bounding box coordinates (x1, y1, x2, y2)

From the bounding box we compute:

width  = x2 - x1
height = y2 - y1
area   = width × height

The bounding box serves as the primary representation of object location and size in the frame.

YOLO was chosen because it provides:

real-time detection speed

single-pass inference

low latency suitable for ADAS systems

Distance Approximation Using Bounding Box Size

In monocular vision systems, actual depth information is not available.
Therefore the system uses bounding box size as an approximation of object distance.

Perspective principle:

object_size ∝ 1 / distance

Meaning:

larger bounding box → object closer to the camera

smaller bounding box → object farther away

The system tracks bounding box area changes over time to estimate whether an object is approaching the vehicle.

Example:

area(t)   = 1200
area(t+1) = 2000

If the area increases significantly, the object is likely approaching.

Position-Based Proximity Estimation

Object position within the frame also provides distance information.

For vehicle-mounted cameras:

objects closer to the vehicle appear lower in the frame

objects farther away appear near the top of the frame

The system computes the vertical center of the object:

center_y = (y1 + y2) / 2

Interpretation:

higher center_y → object closer

lower center_y → object farther

This heuristic helps estimate proximity when combined with bounding box size.

Motion Analysis

To determine whether an object is moving toward the vehicle, the system performs motion analysis.

For lateral motion detection the system uses Farneback Optical Flow.

Optical flow estimates pixel movement between frames.

If a pixel moves from:

(x, y) → (x + dx, y + dy)

its motion vector is:

v = (dx, dy)

By analyzing these vectors, the system determines whether an object:

moves toward the vehicle

moves away

moves parallel to the vehicle

This is especially important for detecting vehicles entering blind spots.

Camera-Specific Detection Logic

Different cameras use different proximity estimation strategies because the type of threat varies by direction.

Front Camera

Focus: collision risk with objects ahead.

Detection logic:

vertical position in frame

bounding box size

bounding box growth over time

Objects appearing lower in the frame and increasing in size indicate a possible forward collision risk.

Rear Camera

Focus: approaching vehicles from behind.

Detection logic:

bounding box size

bounding box growth

vertical proximity

If an object rapidly increases in size, the system assumes it is approaching the vehicle.

Left Camera

Focus: vehicles approaching the blind spot.

Detection logic:

bounding box growth across frames

object proximity to the vehicle side

The system monitors whether objects gradually increase in size, indicating they are entering the blind spot.

Right Camera

Focus: lateral motion into the vehicle lane.

Detection logic:

optical flow motion vectors

direction of object movement

If motion vectors indicate that an object is moving toward the vehicle path rather than parallel to it, the risk level increases.

Proximity Scoring

The system combines multiple indicators into a proximity score.

Conceptual formula:

score =
w1 × size_factor +
w2 × motion_factor +
w3 × position_factor

Where:

size_factor represents bounding box size

motion_factor represents movement toward the vehicle

position_factor represents object location within the frame

The weights (w1, w2, w3) determine the relative importance of each factor.

Example:

w1 = 0.4
w2 = 0.4
w3 = 0.2
Threat Classification

Based on the computed score, the system classifies the risk level:

SAFE      : no object in danger zone
WARNING   : object detected near the vehicle
DANGER    : object very close or moving toward the vehicle

These classifications are used to trigger visual or audio warnings.

Real-Time Optimization

To maintain real-time performance, several optimizations are used.

Frame Skipping

Instead of processing every frame, the system processes only a subset.

Example:

30 FPS input
↓
10 FPS processed

This significantly reduces computation cost.

Exponential Moving Average (EMA)

EMA is used to smooth FPS and motion metrics:

EMA = α × current + (1 − α) × previous

This prevents sudden fluctuations in system indicators.

Key Concepts Used

The system combines several computer vision techniques:

real-time object detection

bounding box geometry

monocular distance approximation

motion estimation

optical flow analysis

proximity scoring

Together these techniques allow the system to estimate collision risk using only camera input.

Limitations

As a monocular vision-based system, the system has limitations:

cannot measure exact physical distance

performance depends on lighting conditions

detection accuracy depends on object visibility

More advanced ADAS systems often combine camera data with:

radar

lidar

depth sensors

Future Improvements

Possible improvements include:

object tracking (SORT / DeepSORT)

trajectory prediction

camera calibration for better distance estimation

multi-sensor fusion
