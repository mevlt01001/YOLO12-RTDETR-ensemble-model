import onnx
from TorchFiles import *
from models_to_onnx import convert
from snc4onnx import combine
from sor4onnx import rename



yolo10_onnx = onnx.load("models/yolov10l.onnx")
yolo12_onnx = onnx.load("models/yolo12l.onnx")

yolo10_onnx = rename(old_new=["yolo10_image", "image"], onnx_graph=yolo10_onnx)
yolo12_onnx = rename(old_new=["yolo12_image", "image"], onnx_graph=yolo12_onnx)



image_sender_onnx = convert(
    model=image_sender(),
    onnx_name="image_splitter.onnx",
    input_shape=[(1,3,640,640)],    
    input_names=["image"],
    output_names=["image1", "image2"],
    OPSET=19
    )

image_yolo_yolo = combine(
    onnx_graphs=[
        image_sender_onnx, yolo10_onnx, yolo12_onnx
    ],
    op_prefixes_after_merging=[
        "init1", "next1", ""
    ],
    srcop_destop=[
        [image_sender_onnx.graph.output[0].name, yolo10_onnx.graph.input[0].name],
        ["init1_"+image_sender_onnx.graph.output[1].name, yolo12_onnx.graph.input[0].name],
        ],
    output_onnx_file_path="demo_with_yolo10.onnx"
)

combined = rename(old_new=[image_yolo_yolo.graph.input[0].name, "image", ], onnx_graph=image_yolo_yolo)
combined = rename(old_new=[image_yolo_yolo.graph.output[0].name, "yolo10_out"], onnx_graph=combined)
combined = rename(old_new=[image_yolo_yolo.graph.output[1].name, "yolo12_out"], onnx_graph=combined)

onnx.save_model(combined, "demo_with_yolo10.onnx")

yolo12_postprocess = YOLO_postprocess(score_threshold=0.2, iou_threshold=0.55)

yolo12_postprocess_onnx = convert(
    model=yolo12_postprocess,
    onnx_name="yolo12_postprocess",
    input_shape=[
        (1,84,8400)
    ],
    input_names=["yolo_postprocess_in"],
    output_names=["yolo_postprocess_out"],
    OPSET=19
)

ensembled_model = combine(
    onnx_graphs=[
        combined,
        yolo12_postprocess_onnx
    ],
    srcop_destop=[
        ["yolo12_out","yolo_postprocess_in"]
    ],
    output_onnx_file_path="YOLO10-YOLO12_NMS.onnx"
)