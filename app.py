"""Module providingFunction printing python version."""
import sqlite3
import pickle
import streamlit as st
import pandas as pd
st.set_page_config(page_title='ANPR', page_icon='car')


hide_menu_style = '''
<style>
#MainMenu {visibility: hidden;}
'''
st.markdown(hide_menu_style, unsafe_allow_html=True)

conn = sqlite3.connect ('ANPR.db')
c = conn.cursor()

def create_table():
    """Create a Table"""
    c.execute('CREATE TABLE IF NOT EXISTS registered(guard TEXT, plate_number CHAR, name TEXT, gender TEXT, date DATE, reason TEXT)')

def add_data(guard, plate_number, name, gender, date, reason):
    """Add data to the database"""
    c.execute(
        'INSERT INTO registered(guard, plate_number, name, gender, date, reason)VALUES(?,?,?,?,?,?)', (guard, plate_number, name, gender, date, reason))
    conn.commit()
    
def view_all_data():
    """View all data in the database"""
    c.execute('SELECT * FROM registered') 
    data = c.fetchall()
    return data 

def delete_data(guard):
    """Delete Data from the database"""
    c.execute('DELETE FROM registered WHERE Guard="{}"'.format(guard))
    conn.commit()
    
def view_unique_data():
    """Update"""
    c.execute('SELECT DISTINCT name FROM registered')
    data = c.fetchall()
    return data

def edit_data(new_plate_number):
    """Edit plate number"""
    c.execute("UPDATE registered SET ")


###### LOAD MODELS ########
#loaded_model = pickle.load(open(h'C:\Users\Acer\OneDrive\Desktop\final_capstone\object_detection.h5', 'rb'))


col1, col2, col3 = st.columns([8, 10, 7])


def main():
    """A simple CRUD"""

    menu = ["Home", "Registration", "Data"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        col1.caption("Upload an Image")
        col1.file_uploader("Choose File", type=['jpg', 'png', 'jpeg'])
        col2.caption("Data")
        col3.caption("Plate Number")
        
    if choice == "Registration":
        st.title("Register Here")
        create_table()   
        guard = st.selectbox("Guard Name:",["Cyril Amparo", "Joana Mejorada", "Lezly Dagonio"])
        plate_number = st.text_area("Plate Number:")
        name = st.text_input("Name:")
        gender = st.selectbox("Gender:",["Male", "Female"])
        date = st.date_input("Date:")
        reason = st.text_input("Reason:")
        if st.button("Register"):
            add_data(guard, plate_number, name, gender, date, reason)   
            st.success("Successfully {}".format('registered'))
            
    if choice == "Data":
        st.title("Registered Data")
        result = view_all_data()
        df = pd.DataFrame(result, columns=['Guard','Plate Number', 'Name', 'Gender', 'Date', 'Reason'])
        st._legacy_table(df)
        
        unique_data = [i[0] for i in view_all_data()]
        delete_data_by_guard = st.selectbox("Unique Data", unique_data)

        if st.button("Delete"):
            delete_data(delete_data_by_guard)
            st.warning("Deleted: '{}'".format(delete_data_by_guard))

        
if __name__ == '__main__':
    main()
    