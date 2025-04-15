from TorchFiles import NMS_without_score_thresholding
from models_to_onnx import convert

convert(
    model=NMS_without_score_thresholding(iou_threshold=0.55),
    onnx_name="Torchvision_NMS.onnx",
    input_names=["boxes","scores"],
    input_shape=[(8400,4),(8400)],
    output_names=["Slected_boxes_and_scores"],
    OPSET=19
)