import onnx ,os
from snc4onnx import combine
from models_to_onnx import model_to_onnx
from TorchFiles import rtdetr_out_splitter, cxcywh2xyxy, NMS


raw_rtdetr_onnx = onnx.load("models/rtdetr-l.onnx")

rtdetr_out_splitter_onnx = model_to_onnx(
    model=rtdetr_out_splitter(),
    onnx_name="rtdetr_out_splitter_onnx.onnx",
    input_shape=[(1,300,84)],    
    input_names=["splitter_in"],
    output_names=["splitter_cxcywh", "splitter_person_conf"],
    OPSET=19
    )

cxcywh2xyxy_onnx = model_to_onnx(
    model=cxcywh2xyxy(),
    onnx_name="cxcywh2xyxy.onnx",
    input_shape=[(300,4)],    
    input_names=["cxcywh_in"],
    output_names=["xyxy_out"],
    OPSET=19
)

rtdetr_out_boxes_and_xyxy_onnx = combine(
    onnx_graphs=[
        rtdetr_out_splitter_onnx,
        cxcywh2xyxy_onnx
    ],
    op_prefixes_after_merging=["splitter", "box_converter"],
    srcop_destop=[
        ["splitter_cxcywh", "cxcywh_in"],
    ],
    output_onnx_file_path="onnx_folder/rtdetr_out_boxes_and_xyxy.onnx",    
)

nms_onnx = model_to_onnx(
    model=NMS(),
    onnx_name="nms.onnx",
    input_shape=[(300,4), (300)],    
    input_names=["nms_xyxy_in", "nms_person_conf_in"],
    output_names=["out"],
    OPSET=19
)

rtdetr_out_boxes_and_xyxy_nms_onnx = combine(
    onnx_graphs=[
        rtdetr_out_boxes_and_xyxy_onnx,
        nms_onnx
    ],
    srcop_destop=[
        ["xyxy_out", "nms_xyxy_in", "splitter_person_conf", "nms_person_conf_in"],
    ],
    output_onnx_file_path="onnx_folder/rtdetr_out_boxes_and_xyxy_nms.onnx",    
)

rtdetr_postprecessed = combine(
    onnx_graphs=[
        raw_rtdetr_onnx,
        rtdetr_out_boxes_and_xyxy_nms_onnx
    ],
    srcop_destop=[
        [raw_rtdetr_onnx.graph.output[0].name, rtdetr_out_boxes_and_xyxy_nms_onnx.graph.input[0].name],
    ],
    output_onnx_file_path="onnx_folder/rtdetr_postprocessed.onnx",
)