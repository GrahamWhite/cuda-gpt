



INSTALLATION:

1. Create virtual python environment:

	python -m venv <project_name>

2. Activate the virtual environment:

(navigate to Scripts directory in virtual environment)
cd <project_name>/Scripts
activate


3. Install dependancies
	*Notes: 
		Numpy must be of version 1.x.x as shown in command 2
		Torch mush be installed with cuda support as shown in command 3
		Ensure Nvidia CUDA drivers are updated otherwise this will run on the cpu (slower) not the gpu (faster)!!!

	1: pip3 install numpy pylzma scikit-learn httpserver		
	2: pip3 install numpy==1.26.4 
	3: pip3 install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118

	Optional: (
	python -m ipykernel install --user  --name=<project-name> --display-name <project-name>-gpt



Commands:
	Start Model Training: python gpt_v2.py
	Start Http Server (localhost:8000): python -m http.server


STARTING THE HTTP SERVER
1. Install dependancies
	Node
(Note a restart may be required to be able to access npm/npx from CMD on windows as the PATH needs to be updated)

2. 
cd control_panel
npm install chart.js react-chartjs-2
npm start dev
	


PROJECT
Set Up Model

	Implement Chart.js to track training metrics

Generate News Stories via AI
	
	AI Needs to communicate in English
	AI Needs to train on news stories
	
Feed AI the New Stories
Evaluate AI Predictions
