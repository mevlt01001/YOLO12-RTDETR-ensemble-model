from TorchFiles.models import *
from ultralytics import RTDETR,YOLO
import os

os.makedirs("models", exist_ok=True)
os.chdir('models/')
yolo = YOLO('yolo12l.pt').export(format="onnx")
rtdetr = RTDETR('rtdetr-l.pt').export(format="onnx")
os.chdir('..')
