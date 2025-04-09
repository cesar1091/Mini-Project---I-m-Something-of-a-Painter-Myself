create_env:
	@echo "Creating virtual environment..."
	python -m venv .venv
	@echo "Virtual environment created."
	@echo.
	@echo To activate the virtual environment, run:
	@echo .venv\Scripts\activate
	@echo.
	@echo "Installing dependencies..."
	.venv\Scripts\pip install --upgrade pip
	.venv\Scripts\pip install -r requirements.txt
	@echo "Dependencies installed."
	@echo "Virtual environment setup complete."

app:
	@echo "Running the application..."
	.venv\Scripts\python app.py
	@echo "Application stopped."

deploy: create_env app
	@echo "Deployment complete."