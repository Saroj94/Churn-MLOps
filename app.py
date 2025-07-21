from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import CustomerChurnData, ChurnClassifier
from src.pipeline.training_pipeline import TrainingPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """
    def __init__(self, request: Request):
            self.request: Request = request

            self.gender: Optional[object]=None
            self.SeniorCitizen: Optional[object]=None
            self.Partner : Optional[int]=None
            self.Dependents : Optional[object]=None
            self.tenure  : Optional[float]=None
            self.PhoneService : Optional[object]=None
            self.MultipleLines : Optional[object]=None
            self.InternetService : Optional[object]=None
            self.OnlineSecurity : Optional[object]=None
            self.OnlineBackup : Optional[object]=None
            self.DeviceProtection : Optional[object]=None
            self.TechSupport : Optional[object]=None
            self.StreamingTV : Optional[object]=None
            self.StreamingMovies : Optional[object]=None
            self.Contract : Optional[object]=None
            self.PaperlessBilling : Optional[object]=None
            self.PaymentMethod : Optional[object]=None
            self.MonthlyCharges : Optional[float]=None
            self.TotalCharges : Optional[float]=None
                

    async def get_customer_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.gender = form.get("gender")
        self.SeniorCitizen = form.get("SeniorCitizen")
        self.Partner = form.get("Partner")
        self.Dependents = form.get("Dependents")
        self.tenure  = form.get("tenure")
        self.PhoneService = form.get("PhoneService")
        self.MultipleLines = form.get("MultipleLines")
        self.InternetService = form.get("InternetService")
        self.OnlineSecurity = form.get("OnlineSecurity")
        self.OnlineBackup = form.get("OnlineBackup")
        self.DeviceProtection = form.get("DeviceProtection")
        self.TechSupport = form.get("TechSupport")
        self.StreamingTV = form.get("StreamingTV")
        self.StreamingMovies = form.get("StreamingMovies")
        self.Contract = form.get("Contract")
        self.PaperlessBilling = form.get("PaperlessBilling")
        self.PaymentMethod = form.get("PaymentMethod")
        self.MonthlyCharges = form.get("MonthlyCharges")
        self.TotalCharges =  form.get("TotalCharges")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
            "index.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_customer_data()
        CustomerData = CustomerChurnData(
            gender = form.gender,
            SeniorCitizen = form.SeniorCitizen,
            Partner = form.Partner,
            Dependents = form.Dependents,
            tenure  = form.tenure,
            PhoneService = form.PhoneService,
            MultipleLines = form.MultipleLines,
            InternetService = form.InternetService,
            OnlineSecurity = form.OnlineSecurity,
            OnlineBackup = form.OnlineBackup,
            DeviceProtection = form.DeviceProtection,
            TechSupport = form.TechSupport,
            StreamingTV = form.StreamingTV,
            StreamingMovies = form.StreamingMovies,
            Contract = form.Contract,
            PaperlessBilling = form.PaperlessBilling,
            PaymentMethod = form.PaymentMethod,
            MonthlyCharges = form.MonthlyCharges,
            TotalCharges =  form.TotalCharges
                                )
        # Convert form data into a DataFrame for the model
        Customer_df = CustomerData.get_churn_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = ChurnClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=Customer_df)[0]

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = value # "Customer churn" if value==1 else "Customer not churn"
        # "Response-Yes" if value == 1 else "Response-No"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)