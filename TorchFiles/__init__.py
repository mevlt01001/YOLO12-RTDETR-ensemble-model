from TorchFiles.models import *
from ultralytics import RTDETR,YOLO
import os

os.makedirs("models", exist_ok=True)
os.chdir('models/')
yolo = YOLO('yolo12l.pt')
if not os.path.exists("yolo12l.onnx"):
    yolo.export(format="onnx")
rtdetr = RTDETR('rtdetr-l.pt')
if not os.path.exists("rtdetr-l.onnx"):
    rtdetr.export(format="onnx")
os.chdir('..')
