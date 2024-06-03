from functools import lru_cache
import onnxruntime as ort

@lru_cache(maxsize=1)
def get_sign_model():
    ort_session = ort.InferenceSession("./sign_recognition_model.onnx")
    return ort_session

def inference(sgn_video, sgn_video_low, sgn_heatmap, sgn_heatmap_low):
    sign_model = get_sign_model()
    sgn_video_input_name = sign_model.get_inputs()[0].name
    sgn_heatmap_input_name = sign_model.get_inputs()[1].name
    sgn_video_low_input_name = sign_model.get_inputs()[2].name
    sgn_heatmap_low_input_name = sign_model.get_inputs()[3].name
    decode_output_name = sign_model.get_outputs()[2].name
    
    result = sign_model.run([decode_output_name], {
        sgn_video_input_name: sgn_video.numpy(),
        sgn_heatmap_input_name: sgn_heatmap.numpy(),
        sgn_video_low_input_name: sgn_video_low.numpy(),
        sgn_heatmap_low_input_name: sgn_heatmap_low.numpy()
    })
    
    return result[0]