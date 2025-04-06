import onnx,os
from onnxsim import simplify
from snc4onnx import combine
from models_to_onnx import convert
from TorchFiles import YOLO_postprocess


raw_yolo_onnx = onnx.load("models/yolo12l.onnx")

YOLO_postprocess_onnx = convert(
    model=YOLO_postprocess(),
    onnx_name="YOLO_postprocess.onnx",
    input_shape=[(1,84,8400)],    
    input_names=["yolo_raw_out"],
    output_names=["boxes_and_scores"],
    OPSET=19
)

YOLO12_postprocessed = combine(
    onnx_graphs=[
        raw_yolo_onnx,
        YOLO_postprocess_onnx
    ],
    srcop_destop=[
        [raw_yolo_onnx.graph.output[0].name, YOLO_postprocess_onnx.graph.input[0].name]
    ],
    output_onnx_file_path="onnx_folder/YOLO12_postprocessed.onnx"
)

os.remove("onnx_folder/YOLO_postprocess.onnx")
YOLO12_postprocessed = onnx.shape_inference.infer_shapes(YOLO12_postprocessed)
YOLO12_postprocessed, check  = simplify(YOLO12_postprocessed)
print(f"Simplified: {check}")
onnx.save(YOLO12_postprocessed, "onnx_folder/YOLO12_postprocessed.onnx")