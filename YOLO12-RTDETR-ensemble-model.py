import onnx ,os
from snc4onnx import combine
from sor4onnx import rename
from models_to_onnx import model_to_onnx
from TorchFiles import image_sender, yolo_out_splitter, rtdetr_out_splitter, cxcywh2xyxy, NMS


raw_yolo_onnx = onnx.load("models/yolo12l.onnx")
raw_yolo_onnx = rename(old_new=["images", "yolo_in"], onnx_graph=raw_yolo_onnx)
raw_yolo_onnx = rename(old_new=["output0", "yolo_out"], onnx_graph=raw_yolo_onnx)

raw_rtdetr_onnx = onnx.load("models/rtdetr-l.onnx")
raw_rtdetr_onnx = rename(old_new=["images", "rtdetr_in"], onnx_graph=raw_rtdetr_onnx)
raw_rtdetr_onnx = rename(old_new=["output0", "rtdetr_out"], onnx_graph=raw_rtdetr_onnx)

image_sender_onnx = model_to_onnx(
    model=image_sender(),
    onnx_name="image_splitter.onnx",
    input_shape=[(1,3,640,640)],    
    input_names=["image"],
    output_names=["image1", "image2"],
    OPSET=19
    )

# image sender must connect to yolo and rtdetr inputs

yolo_and_rtdetr = combine(
    onnx_graphs=[
        image_sender_onnx,
        raw_yolo_onnx,
        raw_rtdetr_onnx
    ],
    op_prefixes_after_merging=[
        "sender",
        "yolo",
        "rtdetr",
    ],
    srcop_destop=[
        ["image1", "yolo_in"],
        ["sender_image2", "rtdetr_in"],
    ],
    output_onnx_file_path="onnx_folder/yolo_and_rtdetr.onnx",
)