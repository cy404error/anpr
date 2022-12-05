"""Module providingFunction printing python version."""
import sqlite3
import streamlit as st
import pandas as pd
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
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

        if st.button("recognise"):
            result_img = img_ori= files.upload()
img_ori = cv2.imread('IMG_3389.PNG')
height, width, channel = img_ori.shape
hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)
gray = hsv[:,:,2]
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
#CONTOUR
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_thresh = cv2.adaptiveThreshold(
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
)
#CONTOUR
contours, _ = cv2.findContours(
    img_thresh, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
#
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)
        
# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
#
MAX_DIAG_MULTIPLYER = 5 # 5
MAX_ANGLE_DIFF = 12.0 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3 # 3

def find_chars(contour_list):
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    
result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
#####

PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    
    img_cropped = cv2.getRectSubPix(
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )
    
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue
    
    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
  ###

longest_idx, longest_text = -1, 0
plate_chars = []

for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # find contours again (same as above)
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h

        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h
                
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
###
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img_result,detail=0,paragraph=True)
    print(result)
    st.image(result_img)
            
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


