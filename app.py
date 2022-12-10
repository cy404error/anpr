"""Module providingFunction printing python version."""
import sqlite3
import streamlit as st
import pandas as pd
from tensorflow import keras 
from tensorflow.python.keras.models import Input
from keras.models import load_model
model = load_model('object_detection.h5')
st.set_page_config(page_title='ANPR', page_icon='car')
####### HIDE MENU AND FOOTER

hide_menu_style = '''
<style>
#MainMenu {visibility: hidden;}
'''
st.markdown(hide_menu_style, unsafe_allow_html=True)

######### CONNECT DB ##############

conn = sqlite3.connect('anpr.db')
c = conn.cursor()

def create_table():
    """Create a Table"""
    c.execute(
        'CREATE TABLE IF NOT EXISTS registered(guard TEXT, name TEXT, gender TEXT, date DATE, reason TEXT)')


def add_data(guard, name, gender, date, reason):
    """Add Data"""
    c.execute('INSERT INTO registered(guard, name, gender, date, reason) VALUES(?,?,?,?,?)',
              (guard, name, gender, date, reason))
    conn.commit()

def view_all_data():
    """View all Data"""
    c.execute('SELECT * FROM registered')
    data = c.fetchall()
    return data


def delete_data(Guard):
    """Delete data"""
    c.execute('DELETE FROM registered WHERE Guard="{}"'.format(Guard))
    conn.commit()


def view_unique_data():
    """Update"""
    c.execute('SELECT DISTINCT name FROM registered')
    data = c.fetchall()
    return data


def get_data(Name):
    """Get data to the database"""
    c.execute('SELECT * FROM registered WHERE Name="{}"'.format(Name))
    #c.execute('SELECT * FROM registered WHERE Name=?',(Name))
    data = c.fetchall()
    return data


def edit_data(new_guard, new_name, new_gender, new_date, new_reason):
    """Edit data"""
    c.execute("UPDATE registered SET guard=?, name=?, gender=?, date=?, reason=? WHERE guard=? and name=? and gender=? and date=? and reason=?",
              (new_guard, new_name, new_gender, new_date, new_reason))
    conn.commit()
    data = c.fetchall()
    return data

col1, col2, col3 = st.columns([8, 10, 7])

def main():
    """A simple CRUD"""

    menu = ["Home", "Registration", "Data"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        col1.caption("Upload an Image")
        image_file = col1.file_uploader("Choose File", type = ['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = (image_file)
            
        
        if st.button("recognise"):
           def object_detection (our_image):
            st.image(our_image)


        col2.caption("Data")
        col3.caption("Plate Number") 
        
    if choice == "Registration":
        st.title("Register Here")
        create_table()   
        guard = st.selectbox("Guard Name:",["Cyril Amparo", "Joana Mejorada", "Lezly Dagonio"])
        name = st.text_input("Name:")
        gender = st.selectbox("Gender:",["Male", "Female"])
        date = st.date_input("Date:")
        reason = st.text_input("Reason:")
        if st.button("Register"):
            add_data(guard,name, gender, date, reason)
            st.success("Successfully {}".format('registered'))
    if choice == "Data":
        st.title("Registered Data")
        result = view_all_data()
        df = pd.DataFrame(result,columns=['Guard','Name', 'Gender', 'Date', 'Reason'])
        st._legacy_table(df)

        ########### DELETE #############33

        unique_data = [i[0] for i in view_all_data()]
        delete_data_by_guard = st.selectbox("Unique Data", unique_data)

        if st.button("Delete"):
            delete_data(delete_data_by_guard)
            st.warning("Deleted: '{}'".format(delete_data_by_guard))

        ############# UPDATE ################

        list_of_data = [i[0] for i in view_unique_data()]
        selected_task = st.selectbox("Task to Edit", list_of_data)
        selected_result = get_data(selected_task)
        st.write(selected_result)
        if selected_result:
            guard = selected_result[0][0]
            name = selected_result[0][1]
            gender = selected_result[0][2]
            date = selected_result[0][3]
            reason = selected_result[0][4]

            new_guard = st.text_input("Guard:", guard)
            new_name = st.text_input("Name:",name)
            new_gender = st.selectbox(gender,["Male", "Female"])
            new_date = st.date_input(date)
            new_reason = st.text_input("reason", reason)
            if st.button("Update"):
                edit_data(new_guard,new_name, new_gender, new_date, new_reason)
                st.success("Updated ::{} To{}".format(name, new_name))


if __name__ == '__main__':
    main()


