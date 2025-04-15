import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

# Girişler
boxes = network.add_input(name="boxes", dtype=trt.float32, shape=(1, 8400, 4))
scores = network.add_input(name="scores", dtype=trt.float32, shape=(1, 8400, 1))
max_output_boxes = network.add_input(name="max_output_boxes", dtype=trt.int32, shape=())
iou_threshold = network.add_input(name="iou_threshold", dtype=trt.float32, shape=())
score_threshold = network.add_input(name="score_threshold", dtype=trt.float32, shape=())

# INMSLayer
nms_layer = network.add_nms(boxes, scores, max_output_boxes)
nms_layer.bounding_box_format = trt.BoundingBoxFormat.CENTER_SIZES

nms_layer.set_input(0, boxes)
nms_layer.set_input(1, scores)
nms_layer.set_input(2, max_output_boxes)
nms_layer.set_input(3, iou_threshold)
nms_layer.set_input(4, score_threshold)

# Çıktılar
nms_layer.get_output(0).name = "selected_indices"
network.mark_output(nms_layer.get_output(0))

# Engine oluştur
engine = builder.build_serialized_network(network, config)
os.makedirs("EngineFolder", exist_ok=True)
with open("EngineFolder/INMSLayer.engine", "wb") as f:
    f.write(engine)

