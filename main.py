import os
import shutil
import threading
from datetime import datetime
from wsgiref import simple_server
from flask import Flask, render_template, request, url_for
from flask_cors import cross_origin, CORS
from CCSCommonTasks.CCSDataInjestionComplete import CCSDataInjestionComplete
from CCSTraining.CCSTrainingPipeline import CCSTrainingPipeline
from CCSPrediction.CCSPredictionPipeline import CCSPredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def ccs_home_page():
    img_url = url_for('static', filename='ineuron-logo.webp')
    return render_template('index.html', image_url=img_url)


@app.route("/train", methods=["POST"])
@cross_origin()
def ccs_train_route():
    img_url = url_for('static', filename='ineuron-logo.webp')
    try:
        if request.form is not None:
            file_item = request.files["dataset"]
            if file_item.filename:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H%M%S")
                file_name = f"Concrete_Data_{dt_string}.xls"

                if os.path.isdir("CCSUploadedFiles"):
                    shutil.rmtree("CCSUploadedFiles")
                    os.mkdir("CCSUploadedFiles")
                else:
                    os.mkdir("CCSUploadedFiles")

                with open(os.path.join("CCSUploadedFiles", file_name), 'wb') as f:
                    f.write(file_item.read())

                train_injestion_obj = CCSDataInjestionComplete(is_training=True, data_dir="CCSUploadedFiles")
                train_injestion_obj.ccs_data_injestion_complete()

                if os.path.isdir("CCSModels"):
                    shutil.rmtree("CCSModels")
                    os.mkdir("CCSModels")
                else:
                    os.mkdir("CCSModels")

                training_pipeline = CCSTrainingPipeline()
                t2 = threading.Thread(target=training_pipeline.ccs_model_train)
                t2.start()
                print("AFTER THREAD T2 STARTED")

                return render_template('train.html',
                                       message="Dataset validated.Training Started. Visit the web Application after "
                                               "1hr to perform prediction.", image_url=img_url)

            else:
                message = "No records Found\n TRY AGAIN"
                return render_template("train.html", message=message, image_url=img_url)
    except ValueError as e:
        return render_template('train.html', message=f"ERROR: {str(e)}\n TRY AGAIN", image_url=img_url)

    except KeyError as e:
        return render_template('train.html', message=f"ERROR: {str(e)}\n TRY AGAIN", image_url=img_url)

    except Exception as e:
        return render_template('train.html', message=f"ERROR: {str(e)}\n TRY AGAIN", image_url=img_url)


@app.route('/prediction', methods=["POST"])
@cross_origin()
def ee_prediction_route():
    img_url = url_for('static', filename='ineuron-logo.webp')
    try:
        if request.form is not None:
            file_item = request.files["dataset"]
            if file_item.filename:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H%M%S")
                file_name = f"Concrete_Data_{dt_string}.xls"

                if os.path.isdir("CCSUploadedFiles"):
                    shutil.rmtree("CCSUploadedFiles")
                    os.mkdir("CCSUploadedFiles")
                else:
                    os.mkdir("CCSUploadedFiles")

                with open(os.path.join("CCSUploadedFiles", file_name), 'wb') as f:
                    f.write(file_item.read())

                pred_injestion_obj = CCSDataInjestionComplete(is_training=False, data_dir="CCSUploadedFiles")
                pred_injestion_obj.ccs_data_injestion_complete()

                pred_pipeline = CCSPredictionPipeline()
                result = pred_pipeline.ccs_predict()

                print(result)

                return render_template("predict.html", records=result, image_url=img_url)

            else:
                message = "Using Default CCSPrediction Dataset"

                pred_injestion = CCSDataInjestionComplete(is_training=False, data_dir="CCSPredictionDatasets")
                pred_injestion.ccs_data_injestion_complete()

                pred_pipeline = CCSPredictionPipeline()
                result = pred_pipeline.ccs_predict()

                return render_template("predict.html", message=message, records=result, image_url=img_url)

    except ValueError as e:
        message = f"Value Error: {str(e)}\nTry Again"
        return render_template("predict.html", message=message, image_url=img_url)

    except KeyError as e:
        message = f"Key Error: {str(e)}\nTry Again"
        return render_template("predict.html", message=message, image_url=img_url)

    except Exception as e:
        message = f"Error: {str(e)}\nTry Again"
        return render_template("predict.html", message=message, image_url=img_url)


@app.route("/logs", methods=["POST"])
@cross_origin()
def ee_get_logs():
    img_url = url_for('static', filename='ineuron-logo.webp')
    try:
        if request.form is not None:
            log_type = request.form['log_type']

            with open(os.path.join("CCSLogFiles/", log_type), "r") as f:
                logs = f.readlines()
            return render_template("logs.html", heading=log_type.split("/")[1], logs=logs, image_url=img_url)
        else:
            message = "No logs found"
            return render_template("logs.html", message=message, image_url=img_url)

    except Exception as e:
        message = f"Error: {str(e)}"
        return render_template("logs.html", heading=message, image_url=img_url)


port = int(os.getenv("PORT", 5000))

if __name__ == "__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
