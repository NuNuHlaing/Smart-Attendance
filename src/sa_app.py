# import necessary libaries
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import os
from PIL import Image
from datetime import datetime, timedelta
from tensorflow import keras
from tensorflow.keras.layers import *

# Define Paths
logo_path = "src/sa_logo.png"
data_dict = {0: "Angelina Jolie", 1: "Nu Nu Hlaing", 2: "Will Smith"} # Dictionary mapping prediction index to corresponding names
model_path = "model/smartattendance.h5" # File path to the trained model
csv_file_path = "daily attendance/DataInfo.csv" # File path to the CSV file containing data

# Create face detector
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Set background color for the Streamlit content area
def set_theme():
    dark_theme = """
    <style>
        .stApp {
            background-color: rgb(240, 242, 246);
        }
    </style>
    """
    st.markdown(dark_theme, unsafe_allow_html=True)

# Preprocessing Function to detect, resize, normalize, and save faces from an input image
def preprocess_image(image_array, target_size=(100, 50)):
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Detect a single face in the input image
    faces = face_detector.detectMultiScale(gray, 1.3, 4)

    # Initialize normalized_face as None
    normalized_face = None

    # Loop over all detected faces
    if len(faces) == 1:
        x, y, w, h = faces[0]
        # Crop the face region
        face = image_array[y:y + h + 15, x:x + w + 15]

        # Create a new image with grey background
        grey_background = np.ones_like(face) * 128

        # Get the dimensions of the face region
        face_height, face_width, _ = face.shape

        # Calculate the position to paste the face on the grey background
        paste_x = (grey_background.shape[1] - face_width) // 2
        paste_y = (grey_background.shape[0] - face_height) // 2

        # Paste the face on the grey background
        grey_background[paste_y:paste_y + face_height, paste_x:paste_x + face_width] = face

        # Resize the face to the target size
        resized_face = cv2.resize(grey_background, target_size)

        # Normalize the pixel values to the range [0, 1]
        normalized_face = resized_face.astype(np.float32) / 255.0

    elif len(faces) == 0:
        print("No face detected in the image.")
    else:
        print("Multiple faces detected in the image. Please provide an image with a single face.")
    return normalized_face

# Get detected face and label from Images
def detect_image(image_array, model): 
    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Detect faces using the cascade classifier
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Preprocess the image using the modified function
    preprocessed_image = preprocess_image(image_array)
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)    
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Process detected face
    for (x, y, w, h) in faces:
        # Predict
        prediction = model.predict(np.expand_dims(image_rgb, axis=0))
        maxindex = int(np.argmax(prediction))
        pred_label = data_dict[maxindex]

        # Draw the bounding box on the frame
        cv2.rectangle(image_array, (x - 15, y - 15), (x + w , y + h + 10), (0, 255, 0), 2)
        cv2.putText(image_array, pred_label, (x - 10, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)   
    return pred_label, image_array

# Predict input image and show result
def predict_image_and_show_result(original_image, image_array, model) : 
    # Call the detect_image function
    predicted_label_name, predicted_image_with_rectangles = detect_image(image_array, model)    
    predicted_image_with_rectangles = cv2.cvtColor(predicted_image_with_rectangles, cv2.COLOR_BGR2RGB)
    # Display Uploaded and Predicted images using Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption='Original Image', use_column_width=True)
    with col2:
        st.image(predicted_image_with_rectangles, caption='Predicted Image', use_column_width=True)

    # Show predicted label
    st.markdown("<span style='text-align: center; font-size: 14'>Predicted Name is </span>"
                f"<span style='text-align: center; color: blue; font-size: 14'>{predicted_label_name}</span>", unsafe_allow_html=True)

    # Read CSV and update date time
    st.markdown("<span style='text-align: center; font-size: 14'>Updated Attendance Information : </span>", unsafe_allow_html=True)
    csv_data = read_csv(predicted_label_name) 
    st.dataframe(csv_data)

# Read Data from CSV
def read_csv(predicted_label_name):
    # Read CSV data
    csv_data = pd.read_csv(csv_file_path)

    # Find the rows with the target name and update datetime if predicted_label_name is not None or empty
    if predicted_label_name:
        # Get current datetime
        current_datetime = datetime.now()
        current_date = datetime.now().date()
        # Find the rows with the target name
        target_rows = csv_data[csv_data['Name'] == predicted_label_name]

        # Iterate over target rows
        for idx, row in target_rows.iterrows():
            # If "In" column is empty
            if pd.isnull(row['In']):  
                csv_data.at[idx, 'In'] = current_datetime
            # If "In" column is not empty             
            elif not pd.isnull(row['In']):  
                csv_data.at[idx, 'Out'] = current_datetime

            # Update the "Date" column
            csv_data.at[idx, 'Date'] = current_date

        # Save the updated DataFrame back to the CSV file
        csv_data.to_csv(csv_file_path, index=False)

        # Filter data based on the provided name parameter and show columns for display
        # Use this to show one row corresponding the provided name parameter
        #filtered_data = csv_data[csv_data["Name"] == predicted_label_name]    
            
        # Show all data rows if predicted_label_name is None or empty
        filtered_data = csv_data
    else:
        # Show all data rows if predicted_label_name is None or empty
        filtered_data = csv_data
  
    # Apply the row highlighting function to the filtered DataFrame
    filtered_selected_data = filtered_data.drop(columns=['Date']).style.apply(lambda row: ['background-color: #ADD8E6']*len(row) if row['Name'] == predicted_label_name else ['']*len(row), axis=1)
    return filtered_selected_data

# Home 
def showHomePage() :
    st.markdown("<h4 style='text-align: center;'>using Deep Learning</h4><br><br>", unsafe_allow_html=True) 
    # Cababilities 
    st.markdown("<h5 style='text-align: left;'>Cababilities!</h5>", unsafe_allow_html=True) 

    st.write('- People just need to stand in front of a camera - no special cards or lists.')
    st.write('- Accessible to everyone, regardless of their race, gender and black or white.')
    st.write('- Don\'t save any input data from users.')
    st.write('')
    # Limitations
    st.markdown("<h5 style='text-align: left;'>Limitations!</h5>", unsafe_allow_html=True)         
    st.write('- Sometimes, it doesn\'t work as well if someone looks different.')
    st.write('- May vary depending on lighting conditions, camera quality, and camera angle.')

# Image
def uploadImage(model) : 
    # File uploader allows users to upload images
    uploaded_image = st.file_uploader('Upload your image file here...', type=['jpeg', 'jpg', 'png'])
    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            # Convert the PIL image to a NumPy array
            image_array = np.array(img)            
            # Convert grayscale to RGB if needed
            if image_array.shape[-1] == 1: image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)            
            # Predict input image and show result
            predict_image_and_show_result(uploaded_image, image_array, model)
            cv2.destroyAllWindows() 
        except Exception as e:
            st.write(str(e))
            st.error("Error occurred while processing the image. Please make sure the image format is supported (JPEG, JPG, PNG) and try again.")

# Webcam 
def captureImage(model) :
    # Define custom CSS style
    custom_css = """
    <style>
        .camera-label {
            text-align: left;
            font-size: 18px;
            color: #333;
            margin-bottom: 10px;
        }
        
        .camera-widget {
            border: 2px solid #ccc;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
    """
    # Display the camera input widget with custom CSS classes
    st.markdown(custom_css, unsafe_allow_html=True)
    # Display the labeled camera input
    st.markdown('<div class="camera-label">Please take a photo</div>', unsafe_allow_html=True)
    image = st.camera_input("", key="camera")
    if image is not None:
        st.success("Photo was successfully taken!")
        img = Image.open(image)
        # Convert the PIL image to a NumPy array
        image_array = np.array(img)            
        # Convert grayscale to RGB if needed
        if image_array.shape[-1] == 1: image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB) 
        # Predict input image and show result
        predict_image_and_show_result(image, image_array, model)

# Contact
def contact() :
    st.markdown("<span style='text-align: center; font-size: 14'>You can send email to us via formsubmit.co</span>", unsafe_allow_html=True)
    form_submit = """<form action="https://formsubmit.co/nunuhlaing2011@gmail.com" method="POST">
     <input type="text" name="name" placeholder=" 🙍🏽‍♂️ Name "required>
     <input type="email" name="email" placeholder=" ✉️ Your Email Address">
     <textarea id="subject" name="subject" placeholder=" 📝 Write something here..." style="height:200px"></textarea>
     <input type="hidden" name="_captcha" value="false">
     <button type="submit">Send</button>
     </form>
     <style>
        input[type=text],input[type=email], select, textarea {
        width: 100%;
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
        margin-top: 6px;
        margin-bottom: 16px;
        resize: vertical;
        }
        button[type=submit] 
        {
        background-color: #D1E5F3;
        color: black;
        padding: 12px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        }
        button[type=submit]:hover
        {
        background-color: #2E34DA;
        color = white;
        }
    </style>
     """
    st.markdown(form_submit,unsafe_allow_html=True)

# Save Daily Attendance at 10PM
def save_daily_attendance_timer(): 
    # Set the target time for creating a new CSV
    target_time = datetime.now().replace(hour=9, minute=40, second=0, microsecond=0)  # 10:00 PM
    while True:
        current_time = datetime.now()
        if current_time >= target_time:            
            # Get the current datetime
            current_datetime = datetime.now().date().strftime("%Y%m%d")
            # Get the current date for the new CSV file name
            new_csv_filename = f"DailyAttendance_{current_datetime}.csv"  
            # save daily attendance to new CSV
            save_daily_attendance_to_newCSV(new_csv_filename)  
            # Reset the target time for the next day
            target_time += timedelta(days=1)
        # Calculate the time remaining until the next target time
        time_remaining = (target_time - current_time).total_seconds()        
        # Sleep until the next iteration, ensuring a non-negative sleep time
        time.sleep(max(1, time_remaining))  # Sleep at least 1 second to avoid negative time_remaining
       
# Save Daily Attendance with button click
def save_daily_attendance_Btn() :
    # Clear the main area
    st.empty()
    # Button to save daily attendance
    if st.sidebar.button("Save Daily Attendance"): 
        # Get the current datetime
        current_datetime = datetime.now().date().strftime("%Y%m%d")
        # Get the current date for the new CSV file name
        new_csv_filename = f"DailyAttendance_{current_datetime}.csv"  
        # save daily attendance to new CSV
        save_daily_attendance_to_newCSV(new_csv_filename)  

# Save Daily Attendance to new CSV
def save_daily_attendance_to_newCSV(new_csv_filename) :    
    st.empty() # Clear the main area    
    old_csv_data = pd.read_csv(csv_file_path) # Read the original CSV file
    new_csv_file_path = "daily attendance/" + new_csv_filename
    # Write the data to a new CSV file
    try:
        with open(new_csv_file_path, 'w') as f:            
            f.write('\n') # Write an empty line to create the CSV file        
        old_csv_data.to_csv(new_csv_file_path, index=False) # Copy data to new CSV
        st.success(f"Save Successfully! '{new_csv_file_path}'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    # Clear the "Date", "In" and "Out" columns in the old CSV
    old_csv_data['Date'] = ''
    old_csv_data['In'] = ''
    old_csv_data['Out'] = ''
    old_csv_data.to_csv(csv_file_path, index=False)

# Create the Streamlit app
def main():
    # Set Theme entire streamlit area
    set_theme()
    # Show Logo image
    st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;            
        }
    </style>
    """, unsafe_allow_html=True
    )
    image = Image.open(logo_path)
    st.sidebar.image(image, caption='Smart Attendance')
    # Title 
    st.markdown("<h1 style='text-align: center;'>Smart Attendance</h1>", unsafe_allow_html=True) 
    
    # Choose options: Upload Image or Webcam
    options = ['Select here...', 'Image', 'Webcam', 'Contact Us']
    inputResource = st.sidebar.selectbox('How would you like to be detected?', options, index=options.index('Select here...'))
    
    # Load the smart attendance face detection model
    model = keras.models.load_model(model_path)

    # HomePage of Streamlit App
    if inputResource == 'Select here...':
        # Clear the main area
        st.empty()
        # Show Home Page
        showHomePage()

    elif inputResource == 'Image':
        # Clear the main area
        st.empty()
        # Title of Option 'Image'
        st.markdown("<h4 style='text-align: center;'>Face Detection from Images</h4><br><br>", unsafe_allow_html=True) 
        # Upload Image
        uploadImage(model)

    elif inputResource == 'Webcam':
        # Clear the main area
        st.empty()        
        # Title of Option 'Webcam'
        st.markdown("<h4 style='text-align: center;'>Face Detection from Webcam</h4><br><br>", unsafe_allow_html=True) 
        # WebCam
        captureImage(model)

    elif inputResource == 'Contact Us':
        # Clear the main area
        st.empty()        
        # Title of Contact Us
        st.markdown("<h4 style='text-align: center;'>Contact Us</h4><br><br>", unsafe_allow_html=True) 
        # Contact Us
        contact()

    # daily attendance with timer
    # save_daily_attendance_timer()
    # daily attendance with button click 
    save_daily_attendance_Btn()        

# Load the Streamlit app
if __name__ == '__main__':
    # load main
    main()