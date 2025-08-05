# 🎓 Gradio Seminar Demos

This repository contains 4 interactive [Gradio](https://gradio.app/) demo projects developed for a seminar on machine learning UI prototyping. Each demo showcases a different application of Gradio — from simple image processing to large language model chatbots.

---

## 📁 Demos

| File Name         | Description                                               | GPU Required |
|------------------|-----------------------------------------------------------|-------------- |
| `flip.py`        | Simple demo that flips a text and an image horizontally   | ❌ No         |
| `ml_chart.py`    | ML model on Titanic dataset with data visualization       | ❌ No         |
| `chatbot.py`     | LLM-powered chatbot using Hugging Face Transformers       | ❌ No         |
| `generate_img.py`| BigGAN-based image generator using class vector input     | ✅ Yes (GPU)  |

---

## 🖥️ Run Locally (CPU)

### ✅ Setup (Local)

python -m venv ENV
source ENV/bin/activate
pip install -r requirements_local.txt



## ▶️ Running the Demos

Once you've installed the requirements, run any of the `.py` files using Python:

python flip.py

Gradio will automatically start a web server and print output like:

Running on local URL:  http://127.0.0.1:7860

🌐 Access the Interface
Open your web browser.

Go to the address: http://127.0.0.1:7860

You’ll see the interactive Gradio app running.

⚠️ If you're running multiple demos at the same time, they will use different ports (e.g., 7861, 7862, etc.)

## 🚀 Run on Compute Cluster (with GPU)

### ⚙️ Setup Environment on Cluster

```bash
virtualenv --no-download ENV
source ENV/bin/activate
pip install -r requirements_nibi.txt

📝 Submit Job (SLURM)
Use a job submission script (e.g., job.sh) to launch the desired demo.
sbatch job.sh

🌐 Access Gradio Interface via Port Forwarding
To view the Gradio interface running on the cluster, use SSH tunneling from your local machine:

🖥️ On Your Local Machine
ssh -L 7860:localhost:7860 <username>@nibi.alliancecan.ca
Then, after logging in to the cluster’s login node:

🔄 From Login Node to Compute Node
Once your job is running on a compute node (e.g., node123), forward the port again:

ssh -L 7860:localhost:7860 <username>@node123
🔁 Replace <username> with your actual cluster username.

Now, open http://127.0.0.1:7860 in your local browser to interact with the Gradio app running on the cluster.

🛑 Stop the App
To stop the app, press Ctrl + C in the terminal where the script is running.

