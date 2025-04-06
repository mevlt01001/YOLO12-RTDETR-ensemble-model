import onnx ,os
from onnxsim import simplify
from snc4onnx import combine
from models_to_onnx import convert
from TorchFiles import RTDETR_postprocess

raw_rtdetr_onnx = onnx.load("models/rtdetr-l.onnx")

RTDETR_postprocess_onnx = convert(
    model=RTDETR_postprocess(score_threshold=0.4, iou_threshold=0.5),
    onnx_name="RTDETR_postprocess.onnx",
    input_shape=[(1,300,84)],    
    input_names=["rtdetr_raw_out"],
    output_names=["boxes_and_scores"],
    OPSET=19
)

RTDETR_postprocessed = combine(
    onnx_graphs=[
        raw_rtdetr_onnx,
        RTDETR_postprocess_onnx
    ],
    srcop_destop=[
        [raw_rtdetr_onnx.graph.output[0].name, RTDETR_postprocess_onnx.graph.input[0].name]
    ],
    output_onnx_file_path="onnx_folder/RTDETR_postprocessed.onnx"
)

os.remove("onnx_folder/RTDETR_postprocess.onnx")
RTDETR_postprocessed = onnx.shape_inference.infer_shapes(RTDETR_postprocessed)
RTDETR_postprocessed, check  = simplify(RTDETR_postprocessed)
print(f"Simplified: {check}")
onnx.save(RTDETR_postprocessed, "onnx_folder/RTDETR_postprocessed.onnx")