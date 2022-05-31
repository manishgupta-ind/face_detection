from helper import *
import streamlit as st
import matplotlib.pyplot as plt
import time
import os
from zipfile import ZipFile

sns.set_theme(style="darkgrid")
sns.set()
st.set_page_config(page_title='Face Detection using AI Technique', page_icon='🖖')

st.title('Face Detection in Image Files')

# User Authentication
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    # Main app starts here

    folderPath = st.text_input('Enter folder path which contains all input images:')
    if len(folderPath) != 0:
        files = [file for file in os.listdir(folderPath) if not file.startswith('.')] # Ignore hidden files
        st.write("We are processing {} image files in above folder to extract all faces. Please wait.".format(len(files)))

        total_faces = 0
        # Create a ZipFile Object
        with ZipFile('faces.zip', 'w') as zipObj:
            i=0
            for image in files:
                i += 1
                # read image file
                pixels = plt.imread(folderPath+image)
                faces = face_detection(pixels)
                extracted_faces = faces.get_faces()
                j=0
                # Add multiple files to the zip
                for face in extracted_faces:
                    j += 1
                    total_faces += 1
                    savefile = "Image_" + str(i) + "_face_" + str(j) + ".jpeg"
                    plt.imsave(savefile,face)
                    zipObj.write(savefile)
                    os.remove(savefile)

        st.write("Success!!! Total {} faces extracted from {} image files.".format(total_faces, len(files)))

        with open("faces.zip", "rb") as fp:
            btn = st.download_button(
                label="Download extracted faces",
                data=fp,
                file_name="faces.zip",
                mime="application/zip"
                )
        os.remove('faces.zip')
