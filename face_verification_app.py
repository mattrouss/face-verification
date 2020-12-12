from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty

import cv2
import numpy as np


PROTOTXT_PATH = 'data/deploy.prototxt'
RESNET_MODEL_PATH = 'data/res10_300x300_ssd_iter_140000_fp16.caffemodel'

Builder.load_string('''
<FileChoosePopup>
    title: "Choose a file"
    size_hint: .9, .9
    auto_dismiss: False
    BoxLayout:
        orientation: "vertical"
        FileChooser:
            id: filechooser
            FileChooserListLayout
        BoxLayout:
            size_hint: (1, 0.1)
            pos_hint: {'center_x': .5, 'center_y': .5}
            spacing: 20
            Button:
                text: "Cancel"
                on_release: root.dismiss()
            Button:
                text: "Load"
                on_release: root.load(filechooser.selection)
                id: ldbtn
                disabled: True if filechooser.selection==[] else False
<DisplayLayout>
    orientation: 'vertical'
    size_hint: 1, 1
''')


class LiveCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(LiveCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, RESNET_MODEL_PATH)

            (h, w) = frame.shape[:2]
            # Resize and normalize image to feed to resnet
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            # Use only first detection
            confidence = detections[0, 0, 0, 2]
            if confidence > 0.9:
                box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (255, 0, 0), 2)

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture

class ImageLabel(BoxLayout):
    def __init__(self, image, label, **kwargs):
        super(ImageLabel, self).__init__(**kwargs)
        self.orientation = "vertical"

        if not image:
            image = np.zeros((300, 300, 3), dtype=np.uint8)

        self.image = Image()
        self.update_image(image)

        self.label = Label(text=label)
        
        self.add_widget(self.image)
        self.add_widget(self.label)

       
    def update_image(self, image):
        image_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        buf = image.tostring()
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.image.texture = image_texture


class ReferencePanel(GridLayout):
    def __init__(self, **kwargs):
        super(ReferencePanel, self).__init__(**kwargs)
        self.cols = 1

        title = Label(text='Face Verification\nApp', font_size="30sp", halign='center')
        self.add_widget(title)

        self.cur_face = ImageLabel(None, 'Current face')
        self.ref_face = ImageLabel(None, 'Reference face')

        box = BoxLayout()
        box.add_widget(self.cur_face)
        box.add_widget(self.ref_face)

        self.add_widget(box)
        self.add_widget(Button(text='Hello'))

    def update_current(self, image):
        self.cur_face.update_image(image)

    def update_ref(self, image):
        self.ref_face.update_image(image)


class DisplayLayout(BoxLayout):
    def __init__(self, capture, **kwargs):
        super(DisplayLayout, self).__init__(**kwargs)
        self.capture = capture
        self.my_camera = LiveCamera(capture=capture, fps=30, size_hint=(1, 1))

        side_panel = ReferencePanel()

        self.add_widget(self.my_camera)
        self.add_widget(side_panel)


class FileChoosePopup(Popup):
    load = ObjectProperty()


class MyLayout(GridLayout):
    def __init__(self, capture, **kwargs):
        super(MyLayout, self).__init__(**kwargs)
        
        self.cols = 2

        self.camera = LiveCamera(capture, fps=30)
        self.ref_panel = ReferencePanel()

        button_color = (70 / 255, 127 / 255, 208 / 255, 1)
        screenshot_button = Button(text='Take a picture!',
                                   background_color=button_color,
                                   size_hint_y = None,
                                   height=100)
        screenshot_button.bind(on_press= lambda a:print("picture taken"))

        picture_select_button = Button(text='Choose a picture!',
                                       background_color=button_color,
                                       size_hint_y = None,
                                       height=100)
        picture_select_button.bind(on_press= lambda a:self.open_popup())
        
        self.add_widget(self.camera)
        self.add_widget(self.ref_panel)
        self.add_widget(screenshot_button)
        self.add_widget(picture_select_button)

        self.the_popup = ObjectProperty(None)
        self.file_path = StringProperty("No file chosen")

    def open_popup(self):
        self.the_popup = FileChoosePopup(load=self.load)
        self.the_popup.open()

    def load(self, selection):
        self.file_path = str(selection[0])
        self.the_popup.dismiss()
        print(self.file_path)

        # check for non-empty list i.e. file selected
        if self.file_path:
            img = cv2.imread(self.file_path)
            img = cv2.flip(img, 0)
            self.ref_panel.update_ref(cv2.resize(img, (300, 300)))
            



class FaceVerificationApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        return MyLayout(self.capture)

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()
        pass


if __name__ == '__main__':
    FaceVerificationApp().run()
