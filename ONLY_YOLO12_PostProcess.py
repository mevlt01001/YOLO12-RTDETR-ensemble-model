import onnx,os
from onnxsim import simplify
from snc4onnx import combine
from models_to_onnx import convert
from TorchFiles import YOLO_postprocess_without_score_thresholding

post_process = YOLO_postprocess_without_score_thresholding(iou_threshold=0.55)

YOLO_postprocess_without_score_thresholding_onnx = convert(
    model=post_process,
    onnx_name="YOLO_POSTPROCESS_POSET_11.onnx",
    input_shape=[(1,84,8400)],
    input_names=["RAW_in"],
    output_names=["SELECTED_out"],
    OPSET=11
)
