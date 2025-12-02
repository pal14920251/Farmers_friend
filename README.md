# Farmers_friend
**AI-Driven Agricultural Advisory System for Indian Farmers** <br>
A modular system that provides crop recommendation, irrigation guidance, and plant-disease detection for Indian farmers.

**Overview** <br>
This project focuses on building an accessible, multilingual agricultural advisory system designed for Indian farmers, especially those in regions with low digital literacy or limited connectivity. The system combines machine learning models, computer vision,irrigation logics, weather API integration, and language translation to provide clear and localized recommendations.

Farmers can input soil nutrient values (or can select their location and we will take the required info), upload images of crop leaves, and receive explanations about suitable crops, irrigation methods, and potential plant diseases—all through a guided interface.


**Objectives** <br>
Provide *simple, localized crop recommendations* based on NPK, pH, temperature, humidity, and rainfall.

Suggest _appropriate irrigation methods_ based on soil type, crop type, and rainfall.

Enable _plant disease identification_ using uploaded leaf images.

Deliver explanations in multiple Indian languages through a translation layer.

Keep the system lightweight and usable even in rural environments with limited connectivity.

**System Components** <br>

**1. Crop Recommendation Model** <br>
Uses a Decision Tree Classifier trained on the Crop Recommendation Dataset.
Features used: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall.
During real-world use, farmers provide soil values, and weather data is fetched through the OpenWeatherMap API.
The model predicts a suitable crop for the given conditions.

**2. Soil Type Identification** <br>
A district-to-soil-zone mapping provides soil type information (e.g., alluvial, black, red).
This is used to refine the irrigation method recommendation.

**3. Irrigation Recommendation** <br>
A rule-based system uses the predicted crop, rainfall level, and soil type to select an irrigation method.
Examples from the project:
      Sandy soil + low rainfall → Drip irrigation
      Clay soil + moderate rainfall → Furrow/Flood irrigation
      High water-demand crops + adequate rainfall → Sprinkler irrigation

**4. Disease Detection using Computer Vision** <br>
Uses the PlantVillage dataset with self-augmentation.
_Preprocessing includes:_ resizing, normalization, flips, rotations.
_Model used:_ ResNet-9
Chosen for its balance of classification performance and computational efficiency.
Training uses Cross-Entropy Loss, Adam optimizer, and One-Cycle learning rate scheduling.
Users upload a leaf image, and the model predicts the disease class.

**5. LLM-Based Explanation Layer** <br>
Once the crop, soil type, irrigation suggestion, and disease result are obtained, they are passed to an LLM.
The LLM generates a **single agricultural advisory message**, combining all results in a clear and coherent way.

**6. Translation Layer (Bhashini)** <br>

The LLM output is translated into the user's chosen language using Bhashini, allowing the final advice to be displayed in a familiar language.

**System Workflow:** <br>
The overall flow is: <br>
1.User selects language and location

2.Inputs soil values

3.Weather API fetches temperature, humidity

4.Soil type retrieved from district mapping

5. Crop recommendation model predicts the crop

6.Irrigation method determined using rule-based logic

7.User uploads leaf image (optional)

8.LLM generates an integrated explanation

9.Translation layer converts result to user’s language

10.Final result shown to the farmer

**Datasets Used:** <br>
**Crop Recommendation Dataset** <br>

**PlantVillage Disease Detection Dataset** <br>
Both datasets are publicly available and used for model training and evaluation.

**Team Contributions:**  <br>
**Aditya Pal**

Developed irrigation recommendation system

Integrated LLM explanation layer

Implemented translation process

Wrote related sections in the final report

**Anumala Sravya**

Processed PlantVillage dataset

Built and trained ResNet-9 disease detection model

Developed inference pipeline

Contributed dataset and model sections

**Urvashi**

Built crop recommendation model

Integrated weather API

Created district-to-soil mapping

Authored crop model and weather integration sections

**Shared Work**

Interface development

Complete report preparation

Final project presentation

**Expected Outcome**

The final system aims to provide farmers with:

1.Clear and localized crop suggestions

2.Practical irrigation guidance

3.Early detection of plant diseases

4.Complete support in their preferred language


