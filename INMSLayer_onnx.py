import onnx
import onnxsim
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

boxes = helper.make_tensor_value_info("Boxes", onnx.TensorProto.FLOAT, [1,8400,4])
scores = helper.make_tensor_value_info("Scores", onnx.TensorProto.FLOAT, [1,1,8400])

max_output_boxes_per_class = helper.make_tensor(
    name="max_output_boxes_per_class",
    data_type=onnx.TensorProto.INT64,
    dims=[],
    vals=[100]
)

iou_threshold = helper.make_tensor(
    name="iou_threshold",
    data_type=onnx.TensorProto.FLOAT,
    dims=[],
    vals=[0.55]
)

score_threshold = helper.make_tensor(
    name="score_threshold",
    data_type=onnx.TensorProto.FLOAT,
    dims=[],
    vals=[0.25]
)

nms_node = helper.make_node(
    "NonMaxSuppression",
    inputs=["Boxes", "Scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
    outputs=["selected_indices"],
    name="NMS",
    center_point_box=1
)

output = helper.make_tensor_value_info("selected_indices", onnx.TensorProto.INT64, [None, 3])

graph = helper.make_graph(
    nodes=[nms_node],
    name="nms_graph",
    inputs=[boxes, scores],
    outputs=[output],
    initializer=[iou_threshold, score_threshold, max_output_boxes_per_class]
)

# Model oluştur
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
model.ir_version = 9
model, check = onnxsim.simplify(model)
onnx.save(model, "onnx_folder/ONNX_NMS.onnx")