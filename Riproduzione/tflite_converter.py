
import tensorflow as tf 
def convert_to_tflite(MODEL_DIR, BATCH_SIZE, STEPS, INPUT_SIZE, modello, nome_modello): 
    run_model = tf.function(lambda x: modello(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], modello.inputs[0].dtype)
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(modello)
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]


    modello.save(MODEL_DIR, save_format="tf", signatures = concrete_func )

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()

    with open(nome_modello+".tflite", "wb") as f:
        f.write(tflite_model)