# this file will show the image


from cmath import e
import PySimpleGUI as sg
import os.path
import PIL
from PIL import Image
import io
import base64
import os
import csv

from prediction import predict_image

# First the window layout in 2 columns


#Declare Variables
THUMBNAIL_SIZE = (200,200)
IMAGE_SIZE = (600,600)
THUMBNAIL_PAD = (1,1)
data = []


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = min(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def convert_to_bytes(file_or_bytes, resize=None, fill=False):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
    if fill:
        img = make_square(img, THUMBNAIL_SIZE[0])
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()


def display_image_window(filename, window):
    try:
        window["-TOUT-"].update("The input file is: {}".format(filename.split('/')[-1]))
        window["-IMAGE-"].update(data=convert_to_bytes(filename, IMAGE_SIZE))
    except Exception as e:
        print('** Display image error **', e)
        return

def save_to_csv_file(data):


    header = ['sample', 'solder area', 'prsistine area', 'solder_ratio']
    

    with open('results.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for sample in data:
            writer.writerow(sample)
    
        





def main():       

    file_list_column = [
        [
            sg.Text("Image Folder"),
            sg.In(size=(25,1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),

        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40,20), key="-FILE LIST-"
            )
        ],
        [
            sg.Button("EXIT")
            ],
        [
            sg.Button("PREDICT")
            ],
        [
            sg.Button("SAVE")
            ],
    ]


    image_viewer_column = [
        [sg.Text("Choose an image from list on left:")],
        [sg.Text(size=(40,1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
        
    ]

    result_viewer_column = [
        [sg.Text("Prediction from Input:")],
        [sg.Text(size=(80,1), key="-RESULT_TOUT-")],
        [sg.Text(size=(80,1), key="-RESULT_TOUT_SR-")],
        [sg.Image(key="-RESULT-")],
        [sg.Button(button_text="Set Pristine", key="-SAVE-PRISTINE-")],
    ]



    # --- Full layout -----

    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
            sg.VSeperator(),
            sg.Column(result_viewer_column),
        ]
    ]

    window = sg.Window("Image Viewer", layout)

    while True:
        event, values =window.read()
        if event == "EXIT" or event == sg.WIN_CLOSED:
            break

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list=os.listdir(folder)
            except:
                file_list=[]

            fnames=[
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".png", ".gif", ".jpg"))
            ]
            window["-FILE LIST-"].update(fnames)

        elif event == "-FILE LIST-": # a file was chosen from the listbox

            
            
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                display_image_window(filename, window)
                #window["-TOUT-"].update(filename)
                #window["-IMAGE-"].update(data=convert_to_bytes(filename, IMAGE_SIZE), enable_events=True)
            except:
                pass
            
        
        elif event == "PREDICT":
            image, count_area = predict_image(filename)

            #check if pristine area exists
            try:
                PRISTINE_AREA 
            except NameError:
                PRISTINE_AREA = count_area 
            
            window["-RESULT_TOUT-"].update("The area of the solder area + transition is {} Pixels".format(count_area))
            window["-RESULT_TOUT_SR-"].update("The Solder Ratio (Solder_Spread/Pristine) is {}".format(count_area/PRISTINE_AREA))
            window["-RESULT-"].update(data=convert_to_bytes(image, IMAGE_SIZE))
            data.append([filename,count_area,PRISTINE_AREA, count_area/PRISTINE_AREA])
        
        elif event == "-SAVE-PRISTINE-":
            try:
                PRISTINE_AREA=count_area
            except:
                print("Please predict image first")
                

        elif event == "SAVE":
            save_to_csv_file(data)





    window.close()   

if __name__ == "__main__":
    main() 
