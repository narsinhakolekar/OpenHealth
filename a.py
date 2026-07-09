import warnings
warnings.simplefilter('ignore')
import uuid
import shutil
import os
import cv2
import sys
import pickle
import subprocess
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, redirect, url_for

from flask import Flask, render_template, request
from OpenHealth.DiseaseDetection.Malaria_Prediction.pipelines.Prediction_pipeline import PredictionPipeline as MalariaPredictionPipeline
from OpenHealth.DiseaseDetection.Diabetes_Disease_Prediction.pipelines.Prediction_pipeline import Diabetes_Data, PredictDiabetes
from OpenHealth.DiseaseDetection.Heart_Disease_Prediction.pipelines.Prediction_pipeline import CustomData, PredictPipeline
from OpenHealth.DiseaseDetection.Breast_Cancer_Prediction.pipelines.Prediction_pipeline import PredictBCancer, BCancer_Data
from OpenHealth.DiseaseDetection.Parkinsons_Disease_Prediction.pipelines.Prediction_pipeline import PredictParkinsons, Parkinsons_Data
from OpenHealth.DiseaseDetection.Liver_Disease_Prediction.pipelines.Prediction_pipeline import LiverData, PredictLiver

# from OpenHealth.DiseaseDetection.Brain_Tumor_Detection.pipelines.Prediction_pipeline import PredictBrainTumour
# kidney_model = load_model(r'Artifacts\Kidney_Disease\Kidney_Model.h5')
# kidney_model = load_model(r'Artifacts\Kidney_Disease\Kidney_Model.h5')

kidney_model = None
kidney_model_path = os.path.join("Artifacts", "Kidney_Disease", "Kidney_Model.h5")
if os.path.exists(kidney_model_path):
    kidney_model = load_model(kidney_model_path)

# ---------------------------------------------------
# LUNG MODEL
# ---------------------------------------------------
# lung_model = None
# lung_model_path = os.path.join("Artifacts", "Lung_Disease", "Lung_Model.h5")
# if os.path.exists(lung_model_path):
#     lung_model = load_model(lung_model_path)
#     print(f"[INFO] Lung model loaded from: {lung_model_path}")
# else:
#     print(f"[WARNING] Lung model not found at: {lung_model_path}")

lung_model = load_model(r'Artifacts\Lung_Disease\Lung_Model.h5')


# =========================
# Brain Tumour model loader
# =========================
brain_model = None

try:
    brain_model_path = os.path.join("Artifacts", "Brain_Tumour", "BrainModel.h5")
    if os.path.exists(brain_model_path):
        brain_model = load_model(brain_model_path)
        print("Brain Tumour model loaded successfully.")
    else:
        print(f"Brain Tumour model not found at: {brain_model_path}")
except Exception as e:
    brain_model = None
    print(f"Brain Tumour model loading failed: {e}")


# Missing model artifacts -> disable for now
# brain_model = load_model(r'Artifacts\Brain_Tumour\BrainModel.h5')
# kidney_model = load_model(r'Artifacts\Kidney_Disease\Kidney_Model.h5')
# livermodel = pickle.load(open(r'Artifacts\Liver_Disease\Liver_Model.pkl', 'rb'))
# from src.utils import gen_from_image, gen_from_text, get_med
# from flask import Flask, render_template,request,redirect,url_for
# from src.Multi_Disease_System.Parkinsons_Disease_Prediction.pipelines.Prediction_pipeline import Parkinsons_Data, PredictParkinsons
# from src.Multi_Disease_System.Breast_Cancer_Prediction.pipelines.Prediction_pipeline import BCancer_Data, PredictBCancer
# from src.Multi_Disease_System.Diabetes_Disease_Prediction.pipelines.Prediction_pipeline import Diabetes_Data, PredictDiabetes
# from src.Multi_Disease_System.Heart_Disease_Prediction.pipelines.Prediction_pipeline import CustomData, PredictPipeline
# brain_model = load_model('Artifacts\Brain_Tumour\BrainModel.h5')
# kidney_model = load_model('Artifacts\Kidney_Disease\Kidney_Model.h5')
# #lung_model = load_model('Artifacts\Lung_Disease\Lung_Model.h5')
# livermodel = pickle.load(open('Artifacts\Liver_Disease\Liver_Model.pkl', 'rb'))
# #liverpreprocessor = pickle.load(open('Artifacts\Liver_Disease\Liver_Preprocessor.pkl', 'rb'))



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Main.html')

# @app.route('/services')
# def index1():
#     try:
#         return render_template('services.html')
#     except:
#         return render_template('error.html')

@app.route('/services')
def index1():
    return render_template('Main.html')
    
'''@app.route('/landing')
def other():
    return render_template('landing.html')'''

@app.route('/chatbot')
def run_streamlit():
    try:
        subprocess.Popen(['streamlit', 'run', 'src/GeminiMed/app.py'])
        return redirect(url_for('index'))
    except:
        return render_template('error.html')


@app.route('/recognition')
def run_streamlit1():
    try:
        subprocess.Popen(['streamlit', 'run', 'src/MedicineRecognition/app.py'])
        return redirect(url_for('index'))
    except:
        return render_template('error.html')

@app.route('/food/<disease>/<tumor_type>', methods=['GET', 'POST'])
def more_info(disease, tumor_type):
    if request.method == 'POST':
        prompt = f"Give me information about the {disease} for this suffering type {tumor_type} in the following paragraph format:\
        Disease Name: \
        Disease Description:\
        Disease Symptoms:\
        Disease Treatment:\
        Disease Food to Eat:\
        Disease Food to Avoid:"  
        # Generate information based on the disease and tumor type
        answer = text_model.generate_content(prompt) # Replace with actual generated content
        ans = answer.replace('*', '\n') 
        return render_template("llm.html", answer=ans)
    return render_template("llm.html", disease=disease, tumor_type=tumor_type)





# @app.route('/bcancer', methods=["GET", "POST"])
# def brain_post():
#     if request.method == 'POST':
#         try:
#             data = BCancer_Data(
#                 texture_mean = float(request.form['texture_mean']),
#                 smoothness_mean = float(request.form['smoothness_mean']),
#                 compactness_mean = float(request.form['compactness_mean']),
#                 concave_points_mean = float(request.form['concave_points_mean']),
#                 symmetry_mean = float(request.form['symmetry_mean']),
#                 fractal_dimension_mean = float(request.form['fractal_dimension_mean']),
#                 texture_se = float(request.form['texture_se']),
#                 area_se = float(request.form['area_se']),
#                 smoothness_se = float(request.form['smoothness_se']),
#                 compactness_se = float(request.form['compactness_se']),
#                 concavity_se = float(request.form['concavity_se']),
#                 concave_points_se = float(request.form['concave_points_se']),
#                 symmetry_se = float(request.form['symmetry_se']),
#                 fractal_dimension_se = float(request.form['fractal_dimension_se']),
#                 texture_worst = float(request.form['texture_worst']),
#                 area_worst = float(request.form['area_worst']),
#                 smoothness_worst = float(request.form['smoothness_worst']),
#                 compactness_worst = float(request.form['compactness_worst']),
#                 concavity_worst = float(request.form['concavity_worst']),
#                 concave_points_worst = float(request.form['concave_points_worst']),
#                 symmetry_worst = float(request.form['symmetry_worst']),
#                 fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
#                 )
#             final_data = data.get_data_as_dataframe()
#             predict_pipeline = PredictBCancer()
#             pred = predict_pipeline.predict(final_data)
#             an = round(pred[0], 2)
#             return render_template('bcancer.html', final_result=an)
#         except:
#             pass
#     return render_template('bcancer.html')
@app.route('/breastcancer', methods=["GET", "POST"])
def breastcancer():
    if request.method == "POST":
        try:
            data = BCancer_Data(
                texture_mean=float(request.form.get("texture_mean")),
                smoothness_mean=float(request.form.get("smoothness_mean")),
                compactness_mean=float(request.form.get("compactness_mean")),
                concave_points_mean=float(request.form.get("concave_points_mean")),
                symmetry_mean=float(request.form.get("symmetry_mean")),
                fractal_dimension_mean=float(request.form.get("fractal_dimension_mean")),
                texture_se=float(request.form.get("texture_se")),
                area_se=float(request.form.get("area_se")),
                smoothness_se=float(request.form.get("smoothness_se")),
                compactness_se=float(request.form.get("compactness_se")),
                concavity_se=float(request.form.get("concavity_se")),
                concave_points_se=float(request.form.get("concave_points_se")),
                symmetry_se=float(request.form.get("symmetry_se")),
                fractal_dimension_se=float(request.form.get("fractal_dimension_se")),
                texture_worst=float(request.form.get("texture_worst")),
                area_worst=float(request.form.get("area_worst")),
                smoothness_worst=float(request.form.get("smoothness_worst")),
                compactness_worst=float(request.form.get("compactness_worst")),
                concavity_worst=float(request.form.get("concavity_worst")),
                concave_points_worst=float(request.form.get("concave_points_worst")),
                symmetry_worst=float(request.form.get("symmetry_worst")),
                fractal_dimension_worst=float(request.form.get("fractal_dimension_worst"))
            )

            final_data = data.get_data_as_dataframe()
            predict_pipeline = PredictBCancer()
            pred = predict_pipeline.predict(final_data)

            result = "Malignant" if int(pred[0]) == 1 else "Benign"
            return render_template("bcancer.html", final_result=result)

        except Exception as e:
            return render_template("bcancer.html", final_result=f"Error: {str(e)}")

    return render_template("bcancer.html")



@app.route('/diabetes', methods=["GET", "POST"])
def diabetes():
    if request.method == "POST":
        try:
            data = Diabetes_Data(
                pregnancies=int(request.form.get("pregnancies")),
                Glucose=int(request.form.get("Glucose")),
                BloodPressure=int(request.form.get("BloodPressure")),
                skin_thickness=int(request.form.get("skin_thickness")),
                insulin=int(request.form.get("insulin")),
                BMI=float(request.form.get("BMI")),
                DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction")),
                Age=int(request.form.get("Age"))
            )

            final_data = data.get_data_as_dataframe()
            predict_pipeline = PredictDiabetes()
            pred = predict_pipeline.predict(final_data)

            result = "Diabetic" if int(pred[0]) == 1 else "Not Diabetic"
            return render_template("diabetes.html", final_result=result)

        except Exception as e:
            return render_template("diabetes.html", final_result=f"Error: {str(e)}")

    return render_template("diabetes.html")

@app.route('/heart', methods=["GET", "POST"])
def heart():

    if request.method == "POST":

        try:

            print(request.form)


            data = CustomData(

                Age=int(request.form.get("Age", 0)),

                Sex=request.form.get("Sex", "M"),

                ChestPainType=request.form.get(
                    "ChestPainType",
                    "ATA"
                ),

                RestingBP=int(
                    request.form.get("RestingBP", 0)
                ),

                Cholesterol=int(
                    request.form.get("Cholesterol", 0)
                ),

                FastingBS=int(
                    request.form.get("FastingBS", 0)
                ),

                RestingECG=request.form.get(
                    "RestingECG",
                    "Normal"
                ),

                MaxHR=int(
                    request.form.get("MaxHR", 0)
                ),

                ExerciseAngina=request.form.get(
                    "ExerciseAngina",
                    "N"
                ),

                Oldpeak=float(
                    request.form.get("Oldpeak", 0)
                ),

                ST_Slope=request.form.get(
                    "ST_Slope",
                    "Up"
                )

            )


            final_data = data.get_data_as_dataframe()


            print("Input Data:")
            print(final_data)



            predict_pipeline = PredictPipeline()


            prediction = predict_pipeline.predict(
                final_data
            )


            print("Prediction:", prediction)



            if prediction[0] == 1:

                result = "⚠️ Heart Disease Risk Detected"

            else:

                result = "✅ No Heart Disease Risk"



            return render_template(
                "heart.html",
                final_result=result
            )


        except Exception as e:

            print("HEART ERROR:", e)

            return render_template(
                "heart.html",
                final_result=f"Error: {e}"
            )


    return render_template("heart.html")

@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        temp_path = None
        try:
            # ---------- Validate uploaded file ----------
            if 'file' not in request.files:
                return render_template(
                    'kidney.html',
                    prediction="No file uploaded"
                )

            file = request.files['file']

            if file.filename == '':
                return render_template(
                    'kidney.html',
                    prediction="No file selected"
                )

            allowed_extensions = {'png', 'jpg', 'jpeg'}
            ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''

            if ext not in allowed_extensions:
                return render_template(
                    'kidney.html',
                    prediction="Invalid file type. Please upload a PNG, JPG, or JPEG image."
                )

            # ---------- Save uploaded file temporarily ----------
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "kidney_input.jpg")
            file.save(temp_path)

            # ---------- Read and preprocess image ----------
            img = cv2.imread(temp_path)

            if img is None:
                return render_template(
                    'kidney.html',
                    prediction="Unable to read the uploaded image."
                )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype("float32")

            # MobileNetV2 preprocessing
            img = img / 127.5 - 1.0
            img = np.expand_dims(img, axis=0)

            # ---------- Prediction ----------
            preds = kidney_model.predict(img, verbose=0)[0]

            # IMPORTANT:
            # This order must match the training class_indices
            class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

            pred_idx = int(np.argmax(preds))
            raw_label = class_labels[pred_idx]
            confidence = round(float(preds[pred_idx]) * 100, 2)

            # ---------- User-friendly result text ----------
            if raw_label == 'Normal':
                result_text = "Normal Kidney CT"
            elif raw_label == 'Cyst':
                result_text = "Kidney Cyst Detected"
            elif raw_label == 'Stone':
                result_text = "Kidney Stone Detected"
            elif raw_label == 'Tumor':
                result_text = "Kidney Tumor Detected"
            else:
                result_text = raw_label

            # ---------- Probability table ----------
            probs = {
                'Cyst': round(float(preds[class_labels.index('Cyst')]) * 100, 2),
                'Normal': round(float(preds[class_labels.index('Normal')]) * 100, 2),
                'Stone': round(float(preds[class_labels.index('Stone')]) * 100, 2),
                'Tumor': round(float(preds[class_labels.index('Tumor')]) * 100, 2)
            }

            return render_template(
                'kidney.html',
                prediction=result_text,
                confidence=confidence,
                probs=probs
            )

        except Exception as e:
            print("Kidney prediction error:", e)
            return render_template(
                'kidney.html',
                prediction=f"Error: {str(e)}"
            )

        finally:
            # ---------- Always remove temp file ----------
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

    return render_template('kidney.html')
# @app.route('/kidney', methods=['GET', 'POST'])
# def kidney():
#     if request.method == 'POST':
#         try:
#             class_labels = {
#                 0: 'Cyst',
#                 1: 'Normal',
#                 2: 'Stone',
#                 3: 'Tumor'
#             }

#             file = request.files.get('file')

#             if file is None or file.filename == '':
#                 return render_template(
#                     'kidney.html',
#                     error='Please upload a kidney CT image.'
#                 )

#             file_path = 'temp_kidney.jpg'
#             file.save(file_path)

#             img = cv2.imread(file_path)
#             img = cv2.resize(img, (150, 150))
#             img = img.astype("float32") / 255.0
#             img = np.expand_dims(img, axis=0)

#             predictions = kidney_model.predict(img)[0]

#             predicted_index = np.argmax(predictions)
#             prediction_label = class_labels[predicted_index]
#             confidence = float(np.max(predictions)) * 100

#             os.remove(file_path)

#             return render_template(
#                 'kidney.html',
#                 prediction=prediction_label,
#                 confidence=round(confidence, 2),
#                 cyst=round(float(predictions[0]) * 100, 2),
#                 normal=round(float(predictions[1]) * 100, 2),
#                 stone=round(float(predictions[2]) * 100, 2),
#                 tumor=round(float(predictions[3]) * 100, 2)
#             )

#         except Exception as e:
#             return render_template(
#                 'kidney.html',
#                 error=f'Error: {str(e)}'
#             )

#     return render_template('kidney.html')


@app.route('/lung', methods=['GET', 'POST'])
def lung():
    classes = [
        'Bacterial Pneumonia',
        'Corona Virus Disease',
        'Normal',
        'Tuberculosis',
        'Viral Pneumonia'
    ]

    if request.method == 'POST':
        temp_path = None
        try:
            # check file
            if 'file' not in request.files:
                return render_template(
                    'lung.html',
                    error='No file part found in request.'
                )

            file = request.files['file']

            if file.filename is None or file.filename.strip() == '':
                return render_template(
                    'lung.html',
                    error='Please select a lung X-ray image.'
                )

            # create temp folder safely
            temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)

            # safe unique filename
            original_name = secure_filename(file.filename)
            ext = os.path.splitext(original_name)[1].lower()

            if ext not in ['.jpg', '.jpeg', '.png']:
                return render_template(
                    'lung.html',
                    error='Only JPG, JPEG, and PNG images are supported.'
                )

            unique_name = f"{uuid.uuid4().hex}{ext}"
            temp_path = os.path.join(temp_dir, unique_name)

            # save uploaded file
            file.save(temp_path)

            # preprocess image
            img = Image.open(temp_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # predict
            prediction = lung_model.predict(img_array, verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            predicted_class = classes[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

            probability_map = {
                classes[i]: round(float(prediction[i]) * 100, 2)
                for i in range(len(classes))
            }

            # delete temp file AFTER prediction
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            return render_template(
                'lung.html',
                prediction=predicted_class,
                confidence=round(confidence, 2),
                probability_map=probability_map
            )

        except Exception as e:
            print("LUNG ERROR:", e)

            # cleanup if file exists
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

            return render_template(
                'lung.html',
                error=f"Error while processing image: {str(e)}"
            )

    return render_template('lung.html')

@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            data = LiverData(
                age=float(request.form.get("age")),
                gender=request.form.get("gender"),
                total_bilirubin=float(request.form.get("total_bilirubin")),
                direct_bilirubin=float(request.form.get("direct_bilirubin")),
                alkaline_phosphotase=float(request.form.get("alkaline_phosphotase")),
                alamine_aminotransferase=float(request.form.get("alamine_aminotransferase")),
                aspartate_aminotransferase=float(request.form.get("aspartate_aminotransferase")),
                total_proteins=float(request.form.get("total_proteins")),
                albumin=float(request.form.get("albumin")),
                albumin_globulin_ratio=float(request.form.get("albumin_globulin_ratio"))
            )

            final_data = data.get_data_as_dataframe()
            predictor = PredictLiver()
            pred = predictor.predict(final_data)

            probability = None
            try:
                probs = predictor.predict_proba(final_data)
                if probs is not None:
                    probability = round(float(probs[0][1]) * 100, 2)
            except:
                probability = None

            if int(pred[0]) == 1:
                result = "Liver Disease Detected"
            else:
                result = "No Liver Disease Detected"

            return render_template(
                'liver.html',
                prediction=result,
                probability=probability
            )

        except Exception as e:
            print("LIVER ERROR:", e)
            return render_template(
                'liver.html',
                prediction=f"Error: {str(e)}"
            )

    return render_template('liver.html')

# =========================
# MALARIA CONFIG
# =========================
MALARIA_UPLOAD_DIR = os.path.join("static", "uploads", "malaria")
os.makedirs(MALARIA_UPLOAD_DIR, exist_ok=True)

malaria_predictor = None
try:
    malaria_predictor = MalariaPredictionPipeline()
    print("[INFO] Malaria prediction pipeline loaded successfully.")
except Exception as e:
    print(f"[WARNING] Malaria predictor not loaded yet: {e}")


def format_malaria_result(predicted_class):
    if predicted_class.lower() == "parasitized":
        return "Malaria Infected (Parasitized)"
    return "Uninfected / No Malaria Detected"


@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    prediction = None
    confidence = None
    probabilities = None
    uploaded_image = None
    error = None

    if request.method == 'POST':
        try:
            if malaria_predictor is None:
                raise ValueError("Malaria model is not loaded yet. Finish training first and restart Flask.")

            if 'image' not in request.files:
                error = "Please upload a malaria cell image."
                return render_template(
                    'malaria.html',
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    uploaded_image=uploaded_image,
                    error=error
                )

            file = request.files['image']

            if file.filename == '':
                error = "Please select an image file."
                return render_template(
                    'malaria.html',
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    uploaded_image=uploaded_image,
                    error=error
                )

            allowed_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in allowed_exts:
                error = "Invalid file type. Please upload PNG, JPG, JPEG, or BMP image."
                return render_template(
                    'malaria.html',
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    uploaded_image=uploaded_image,
                    error=error
                )

            unique_name = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(MALARIA_UPLOAD_DIR, unique_name)
            file.save(save_path)

            uploaded_image = save_path.replace("\\", "/")

            result = malaria_predictor.predict(save_path)

            raw_prediction = result["prediction"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            prediction = format_malaria_result(raw_prediction)

        except Exception as e:
            print("MALARIA ERROR:", e)
            error = f"Error while processing image: {str(e)}"

    return render_template(
        'malaria.html',
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        uploaded_image=uploaded_image,
        error=error
    )
    

@app.route('/parkinsons', methods=["GET", "POST"])
def parkinsons():
    if request.method == 'POST':
        try:
            data = Parkinsons_Data(
                MDVPFO=float(request.form.get("MDVPFO")),
                MDVPFHI=float(request.form.get("MDVPFHI")),
                MDVPFLO=float(request.form.get("MDVPFLO")),
                MDVPJ=float(request.form.get("MDVPJ")),
                RPDE=float(request.form.get("RPDE")),
                DFA=float(request.form.get("DFA")),
                spread2=float(request.form.get("spread2")),
                D2=float(request.form.get("D2"))
            )

            final_data = data.get_data_as_dataframe()
            predict_pipeline = PredictParkinsons()
            pred = predict_pipeline.predict(final_data)

            if pred[0] == 1:
                result = "Parkinson's Detected"
            else:
                result = "No Parkinson's Detected"

            return render_template("parkinsons.html", final_result=result)

        except Exception as e:
            print("Parkinson's prediction error:", e)
            return render_template("parkinsons.html", final_result="Error in prediction")

    return render_template("parkinsons.html")

import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
text_model = genai.GenerativeModel('gemini-pro') 

# =========================================================
# BRAIN TUMOUR ROUTES
# =========================================================

@app.route('/brain', methods=['GET', 'POST'])
def brain():
    if request.method == 'POST':
        try:
            if brain_model is None:
                return render_template(
                    'brain_tumour.html',
                    error="Brain Tumour model is not available. Please train/load the model first."
                )

            file = request.files.get('file')

            if file is None or file.filename == '':
                return render_template(
                    'brain_tumour.html',
                    error="Please upload a brain MRI image."
                )

            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)

            temp_filename = f"brain_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            file.save(temp_path)

            img = Image.open(temp_path).convert("RGB")
            img = img.resize((224, 224))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            preds = brain_model.predict(img, verbose=0)[0]

            class_labels = ['Glioma Tumour', 'Meningioma Tumour', 'No Tumour', 'Pituitary Tumour']
            predicted_index = int(np.argmax(preds))
            prediction_label = class_labels[predicted_index]
            confidence = float(np.max(preds) * 100)

            probability_map = {
                class_labels[i]: round(float(preds[i]) * 100, 2)
                for i in range(len(class_labels))
            }

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return render_template(
                'brain_tumour.html',
                prediction=prediction_label,
                confidence=round(confidence, 2),
                probabilities=probability_map
            )

        except Exception as e:
            print("BRAIN ERROR:", e)
            return render_template(
                'brain_tumour.html',
                error=f"Error while processing image: {str(e)}"
            )

    return render_template('brain_tumour.html')


@app.route('/brain_tumour1')
def brain_tumour1():
    prompt = """Give me information about glioma brain tumour in the following format:
Disease Name:
Disease Description:
Disease Symptoms:
Disease Treatment:
Disease Food to Eat:
Disease Food to Avoid:"""
    answer = text_model.generate_content(prompt)
    ans = answer.text.replace('*', '\n') if hasattr(answer, "text") else str(answer).replace('*', '\n')
    return render_template("llm.html", answer=ans)


@app.route('/brain_tumour2')
def brain_tumour2():
    prompt = """Give me information about meningioma brain tumour in the following format:
Disease Name:
Disease Description:
Disease Symptoms:
Disease Treatment:
Disease Food to Eat:
Disease Food to Avoid:"""
    answer = text_model.generate_content(prompt)
    ans = answer.text.replace('*', '\n') if hasattr(answer, "text") else str(answer).replace('*', '\n')
    return render_template("llm.html", answer=ans)


@app.route('/brain_tumour3')
def brain_tumour3():
    prompt = """Give me information about pituitary brain tumour in the following format:
Disease Name:
Disease Description:
Disease Symptoms:
Disease Treatment:
Disease Food to Eat:
Disease Food to Avoid:"""
    answer = text_model.generate_content(prompt)
    ans = answer.text.replace('*', '\n') if hasattr(answer, "text") else str(answer).replace('*', '\n')
    return render_template("llm.html", answer=ans)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
if __name__ == '__main__':

    print(app.url_map)

    app.run(debug=True, host='0.0.0.0', port=5000)