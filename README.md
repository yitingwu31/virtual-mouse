# virtual-mouse

### Setup
This project requires ```python 3.8``` to run.

If you do not have a python 3.8 installed, consider using ```pipenv``` to run python 3.8 in the virtual environment

[installing pipenv & creating virtual environment](https://www.codingforentrepreneurs.com/blog/install-django-on-mac-or-linux/)
```
pip install opencv-python
pip install mediapipe
pip install -U autopy
```

More on installing [autopy](https://pypi.org/project/autopy/)

### Hand-Tracking with mediapipe
```
python HandTrackingModule.py
```

### Simple Gesture Recognition
```
python FingerCounter.py
```
- Zoom
- Scroll (with right hand)
- Cursor point (with right hand)
- Single image, still a lot of bugs to fix

### Recognition with CVZone
```
pip install cvzone
python Zoom2.py
```
- Zoom
- Need to fix the image flip problem
