import json
import zipfile


SOURCE = "temp_isl_model/models/isl_model.keras"
OUTPUT = "temp_isl_model/models/isl_model_fixed.keras"


def clean_config(obj):
    
    """Remove configuration fields incompatible with Keras 2.15."""

    if isinstance(obj, dict):

        # Remove newer initializer fields
        obj.pop("input_axes", None)
        obj.pop("output_axes", None)
        obj.pop("quantization_config", None)
        obj.pop("loss_scale_factor", None)
        obj.pop("gradient_accumulation_steps", None)

        # Convert newer DTypePolicy representation
        if (
            obj.get("class_name") == "DTypePolicy"
            and isinstance(obj.get("config"), dict)
        ):
            dtype_name = obj["config"].get("name")

            if dtype_name:
                # Replace the serialized policy with the plain dtype string
                return dtype_name

        for key in list(obj.keys()):
            obj[key] = clean_config(obj[key])

        return obj

    elif isinstance(obj, list):
        return [clean_config(item) for item in obj]

    return obj


with zipfile.ZipFile(SOURCE, "r") as zin:

    config = json.loads(zin.read("config.json"))

    # Fix InputLayer
    for layer in config["config"]["layers"]:

        if layer.get("class_name") == "InputLayer":

            layer_config = layer["config"]

            # Remove unsupported field
            layer_config.pop("optional", None)

            # Convert batch_shape → batch_input_shape
            batch_shape = layer_config.pop("batch_shape", None)

            if batch_shape is not None:
                layer_config["batch_input_shape"] = batch_shape

    # Fix the rest of the configuration
    config = clean_config(config)

    # Create repaired model
    with zipfile.ZipFile(OUTPUT, "w") as zout:

        for item in zin.infolist():

            data = zin.read(item.filename)

            if item.filename == "config.json":
                data = json.dumps(config).encode("utf-8")

            zout.writestr(item, data)


print("Fixed model created:")
print(OUTPUT)