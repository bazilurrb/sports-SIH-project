# Sports SIH Project – Pushup Detection  

## 📌 Overview  
This project uses **OpenCV** and **MediaPipe** to detect pushups in real-time via webcam. It can be extended for fitness assessment as part of the SIH requirement.  

---

## ⚙️ Installation  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/bazilurrb/sports-SIH-project.git
   cd sports-SIH-project
2. **Create a virtual environment**
    ```bash
    python -m venv venv
3. **Activate the virtual environment**
**On Windows**
     ```bash
    venv\Scripts\activate
**On Mac/Linux**
    ```bash
    source venv/bin/activate
4. **Install Dependencies**
    ```bash
    pip isntall -r requirements.txt
5. **Usage**
Run the pushup detecction script
    ```bash
    python Pushup.py
6. **Project Structure**
```bash
sports-SIH-project/
│── Pushup.py          # Main script for pushup detection
│── requirements.txt   # List of dependencies
│── README.md          # Project documentation
│── .gitignore         # Ignored files (e.g., venv/)
