import tensorrt
import numpy
import os

onnx_folder = "onnx_folder"
engine_folder = "engine_folder"
os.makedirs(engine_folder, exist_ok=True)

#onnx_files = os.listdir(onnx_folder)

for onnx_file_path in ["onnx_folder/YOLO12_INMSLayer.onnx"]:

    file_name = onnx_file_path.lstrip("onnx_folder/").rstrip(".onnx")

    LOGGER = tensorrt.Logger(tensorrt.Logger.VERBOSE)
    BUILDER = tensorrt.Builder(LOGGER)

    network = BUILDER.create_network(1)
    parser = tensorrt.OnnxParser(network,LOGGER)

    with open(onnx_file_path, "rb") as model:
        success = parser.parse(model.read())
    
    if not success:
        print(f"Failed to parse model {file_name}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError(f"Failed to parse model {file_name}")
    
    config = BUILDER.create_builder_config()
    config.profiling_verbosity = tensorrt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, 3*1024*1024*1024) #4gb

    if BUILDER.platform_has_fast_fp16:
        config.set_flag(tensorrt.BuilderFlag.FP16)
        print(f"FP16 precision setted for {file_name}")

    profile = BUILDER.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, min=(1,3,640,640), opt=(1,3,640,640), max=(1,3,640,640))

    config.add_optimization_profile(profile)

    engine = BUILDER.build_serialized_network(network, config)

    if engine is None: raise RuntimeError(f"Engine serialization has an eror for {file_name}")
    
    with open(os.path.join(engine_folder, file_name+".engine"), "wb") as f:
        f.write(engine)
        print(f"{engine_folder}/{file_name}.engine saved successfully!!")
    