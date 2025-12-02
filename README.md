# 🌾 AI-Powered Agricultural Decision Support System

*A multilingual, accessible AI tool for crop recommendation, irrigation planning, and plant disease detection.*

---

## 📌 Overview

Small and marginal farmers across India often lack access to reliable, localized, and scientific agricultural guidance. Many existing tools are either too generic or too complex, making them inaccessible to low-literacy and low-connectivity regions.

This project builds a **lightweight, multilingual AI decision-support system** that helps farmers make data-driven choices about:

* **Which crop to grow**
* **What irrigation method to use**
* **What disease is affecting their plant**

All recommendations are generated using **machine learning, computer vision, weather APIs, rule-based agronomic logic**, and a **guided user interface**, ensuring ease of use.

---

## 🎯 Project Goals

* Provide *localized, trustworthy* crop and irrigation advice.
* Enable *early disease detection* using computer vision.
* Build a *simple, multilingual UI* that works for real-world Indian farming conditions.
* Integrate AI with *soil health card values, real-time weather data*, and farmer inputs.

---

## 🧱 System Architecture

The system integrates multiple AI modules into a unified workflow:

### 1. **Crop Recommendation Model**

* **Dataset:** Kaggle Crop Recommendation Dataset
* **Model:** Decision Tree Classifier
* **Features:**

  * N, P, K
  * pH
  * Temperature
  * Humidity
  * Rainfall
* **Weather data** (temperature, humidity, rainfall) fetched using **OpenWeatherMap API**.

### 2. **Soil-Type Identification**

* District → soil-zone mapping
* Soil type automatically retrieved based on user-selected district
* Soil type feeds into irrigation logic

### 3. **Irrigation Recommendation Engine**

A rule-based engine informed by agricultural domain knowledge:

* Sandy soil + low rainfall → **Drip irrigation**
* Clay soil + moderate rainfall → **Furrow/Flood irrigation**
* High water-demand crop + adequate rainfall → **Sprinkler**

Irrigation = *function(crop, soil_type, rainfall)*

### 4. **Plant Disease Detection (Computer Vision)**

* **Dataset:** PlantVillage (with basic augmentations)
* **Model:** Transfer-learning using **ResNet-50**, modified with:

  * Pretrained ImageNet backbone
  * Final FC layer replaced with `Linear(2048 → num_classes)`
* **Preprocessing:**

  * Resize → Center-crop (224×224)
  * Normalize (ImageNet mean/std)
  * Random flips & rotations
* **Training Details:**

  * Loss: Cross-Entropy
  * Optimizer: Adam
  * LR Scheduler: StepLR / ReduceLROnPlateau
* **Inference:**

  * Farmer uploads a leaf image → Model outputs the predicted disease class.


### 5. **LLM-Based Expert Explanation Layer**

* Aggregates:

  * Crop recommendation
  * Soil type
  * Irrigation strategy
  * Disease diagnosis
* Generates a natural, unified agricultural advisory.

### 6. **Multilingual Translation (Bhashini)**

* Converts final output to any Indian language selected by the farmer.
* English responses bypass translation.

---

## 🔄 End-to-End Workflow

1. User selects **language** and **district**.
2. Enters **soil nutrient values** (N, P, K, pH).
3. System fetches **weather** via API.
4. Crop Recommendation Model predicts best crop.
5. Soil type is auto-selected based on district.
6. Irrigation engine recommends suitable method.
7. (Optional) User uploads a leaf image → Disease detection model predicts disease.
8. LLM synthesizes a unified advisory.
9. Bhashini translates it to the selected language.
10. Final output shown to the user.

---

## 👥 Team Contributions

### **👤 Member 1 — Aditya Pal**

**Irrigation System, LLM Generation, Translation Layer**

* Designed rule-based irrigation recommendation engine
* Implemented LLM-based explanation layer
* Integrated Bhashini for multilingual output
* Authored report sections: Irrigation, LLM Synthesis, Translation

### **👤 Member 2 — Anumala Sravya**

**Disease Detection Pipeline**

* Dataset prep & augmentation
* Implemented & trained ResNet-9 model
* Built leaf-image upload and inference module
* Authored report sections: Dataset Processing, Model Architecture, Evaluation

### **👤 Member 3 — Urvashi**

**Crop Model, Weather API, Soil Mapping**

* Trained Decision Tree crop recommendation model
* Integrated OpenWeatherMap API
* Built district → soil zone mapping
* Authored report sections: Crop Model, Soil Mapping, Weather Integration

### **🤝 Shared Work (All Members)**

* UI design & integration
* Final report writing
* Project presentation creation & delivery

---

## 🛠️ Technologies Used

* **Machine Learning:** scikit-learn (Decision Tree)
* **Deep Learning:** PyTorch (ResNet-9)
* **Backend / APIs:** Python, OpenWeatherMap API
* **Translation:** Bhashini (Indian languages)
* **Frontend/UI:** Streamlit
* **LLM:** Instructional generation layer (HuggingFace / local LLM)

---

## 🚀 Key Features

* Multilingual guided UI
* Crop recommendation using real weather + soil values
* Rule-based irrigation suggestions
* Deep learning–based disease detection
* Unified agricultural advisory generated using LLM
* Fully integrated translation layer for local languages

---

## 🌱 Impact

This prototype demonstrates how **AI can be adapted to real-world agricultural challenges** in a practical, accessible way for farmers in low-resource contexts. By combining machine learning, computer vision, and a guided multilingual interface, the system supports better decision-making and encourages more resilient farming practices.

---
