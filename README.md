# gesture-mouse-control2

### Brief
Control computer cursor by using a webcam and gestures using geometric methods. No other equipment required.  
Still at early stage -- adding classes to get my job done. Functionality I need, as Python classes is:  
- [x] Adaptive skin detector
- [x] Backprojection-based skin detector
- [x] Try cascade (e.g. Haar) detector
- [ ] Haar detector must output only **zero** or **one** detections. Try creating maybe a heat map to achieve that.
- [x] grabcut as alternative/ enhancer for skin detector? Used as ground truth generator.
- [x] Interactive image cropper to extract a skin sample (MouseRoi.py)
- [x] Lucas-Kanade tracker to track hand features (LkTracker.py)
- [ ] Contour, convex hull finder, gesture estimator
- [x] ~Face removal module~ (not really needed for now)
- [ ] Hand centre estimator. Centroid? Max inscribed circle?
- [ ] Library to interface with cursor 
- [ ] Unit tests

### Algorithm description

* **Haar cascade hand detector**: Provides reliable and real-time tracking. It is a binary detector therefore cannot tell which gesture is used but it has been trained with 7000 positive/ negative images to detect a hand if one of the following gestures is used: (1) clenched fingers, (2) raised index finger, (3) victory sign. Can be used to assist the Lukas-Kanade tracker. The bounding box is always around the palm and not the fingers (to make it more robust).

### Branches
Active branch is `dev`
