



INSTALLATION:

1. Create virtual python environment:

	python -m venv <project_name>

2. Activate the virtual environment:

(navigate to Scripts directory in virtual environment)
cd <project_name>/Scripts
activate


3. Install dependancies
	*Notes: Numpy must be of version 1.x.x, and torch mush be installed with cuda support as shown in command 3

	1: pip3 install numpy pylzma scikit-learn
		
	2: pip3 install numpy==1.26.4 
	3: pip3 install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118

	Optional: (
	python -m ipykernel install --user  --name=<project-name> --display-name <project-name>-gpt


Note * other installations include http.server and scipypackage*(or something like that)

Ensure Nvidia CUDA drivers are updated otherwise this will run on the cpu (slower) not the gpu (faster)!!!




PROJECT
Set Up Model

	Implement Chart.js to track training metrics

Generate News Stories via AI
	
	AI Needs to communicate in English
	AI Needs to train on news stories
	
Feed AI the New Stories
Evaluate AI Predictions
