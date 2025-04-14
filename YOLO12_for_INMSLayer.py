import onnx,os
from onnxsim import simplify
from snc4onnx import combine
from models_to_onnx import convert
from TorchFiles import yolo_out_splitter_for_INMSLayer

raw_yolo_onnx = onnx.load("models/yolo12l.onnx")
splitter = yolo_out_splitter_for_INMSLayer()

YOLO_postprocess_onnx = convert(
    model=splitter,
    onnx_name="YOLO_postprocess.onnx",
    input_shape=[(1,84,8400)],    
    input_names=["yolo_raw_out"],
    output_names=["boxes","scores"],
    OPSET=19
)


for output in YOLO_postprocess_onnx.graph.output:
    print(output.name, output.type.tensor_type.shape)


os.remove("onnx_folder/YOLO_postprocess.onnx")

YOLO12_postprocessed = combine(
    onnx_graphs=[
        raw_yolo_onnx,
        YOLO_postprocess_onnx
    ],
    srcop_destop=[
        [raw_yolo_onnx.graph.output[0].name, YOLO_postprocess_onnx.graph.input[0].name]
    ],
)

YOLO12_postprocessed = onnx.shape_inference.infer_shapes(YOLO12_postprocessed)
YOLO12_postprocessed, check  = simplify(YOLO12_postprocessed)
print(f"Simplified: {check}")
onnx.save(YOLO12_postprocessed, f"onnx_folder/YOLO12_boxes_scores.onnx")

NMS_layer = onnx.load("onnx_folder/INMSLayer.onnx")

final = combine(
    onnx_graphs=[
        YOLO12_postprocessed,
        NMS_layer
    ],
    srcop_destop=[
        [YOLO12_postprocessed.graph.output[0].name, NMS_layer.graph.input[0].name, YOLO12_postprocessed.graph.output[1].name, NMS_layer.graph.input[1].name]
    ],
    output_onnx_file_path="onnx_folder/YOLO12_INMSLayer.onnx"
)

final = onnx.shape_inference.infer_shapes(final)
final, check  = simplify(final)

for input in final.graph.output:
    print(input.name, input.type.tensor_type.shape)


onnx.save(final, f"onnx_folder/YOLO12_INMSLayer.onnx")
print(f"Simplified: {check}")