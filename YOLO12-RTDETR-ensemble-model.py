import os
import onnx
from onnxsim import simplify
from snc4onnx import combine
from sor4onnx import rename
from models_to_onnx import convert
from TorchFiles import image_sender, Ensemble_postprocess


raw_yolo_onnx = onnx.load("models/yolo12l.onnx")
raw_yolo_onnx = rename(old_new=["images", "yolo_in"], onnx_graph=raw_yolo_onnx)
raw_yolo_onnx = rename(old_new=["output0", "yolo_out"], onnx_graph=raw_yolo_onnx)

raw_rtdetr_onnx = onnx.load("models/rtdetr-l.onnx")
raw_rtdetr_onnx = rename(old_new=["images", "rtdetr_in"], onnx_graph=raw_rtdetr_onnx)
raw_rtdetr_onnx = rename(old_new=["output0", "rtdetr_out"], onnx_graph=raw_rtdetr_onnx)

image_sender_onnx = convert(
    model=image_sender(),
    onnx_name="image_splitter.onnx",
    input_shape=[(1,3,640,640)],    
    input_names=["image"],
    output_names=["image1", "image2"],
    OPSET=19
    )

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
)
yolo_and_rtdetr = rename(old_new=[yolo_and_rtdetr.graph.input[0].name, "image"], onnx_graph=yolo_and_rtdetr)
yolo_and_rtdetr = rename(old_new=[yolo_and_rtdetr.graph.output[0].name, "yolo_out"], onnx_graph=yolo_and_rtdetr)
yolo_and_rtdetr = rename(old_new=[yolo_and_rtdetr.graph.output[1].name, "rtdetr_out"], onnx_graph=yolo_and_rtdetr)

os.remove("onnx_folder/image_splitter.onnx")
yolo_and_rtdetr = onnx.shape_inference.infer_shapes(yolo_and_rtdetr)
yolo_and_rtdetr, check  = simplify(yolo_and_rtdetr)
print(f"Simplified: {check}")
onnx.save(yolo_and_rtdetr, "onnx_folder/yolo_and_rtdetr.onnx")

postprocess = Ensemble_postprocess(score_threshold=0.5, iou_threshold=0.7)
Ensemble_postprocess_onnx = convert(
    model=postprocess,
    onnx_name="Ensemble_postprocess.onnx",
    input_shape=[
        (1,84,8400),
        (1,300,84),
    ],    
    input_names=["yolo_in", "rtdetr_in"],
    output_names=["boxes_and_scores"],
    OPSET=19
)

YOLO12_RTDETR_ensemble_model = combine(
    onnx_graphs=[
        yolo_and_rtdetr,
        Ensemble_postprocess_onnx
    ],
    srcop_destop=[
        ["yolo_out", "yolo_in", "rtdetr_out", "rtdetr_in"],
    ],
)

os.remove("onnx_folder/yolo_and_rtdetr.onnx")
os.remove("onnx_folder/Ensemble_postprocess.onnx")
YOLO12_RTDETR_ensemble_model = onnx.shape_inference.infer_shapes(YOLO12_RTDETR_ensemble_model)
YOLO12_RTDETR_ensemble_model, check  = simplify(YOLO12_RTDETR_ensemble_model)
print(f"Simplified: {check}")
onnx.save(YOLO12_RTDETR_ensemble_model, f"onnx_folder/YOLO12-RTDETR_ensemble_model_{postprocess.NMS.score_threshold}_{postprocess.NMS.iou_threshold}.onnx")
