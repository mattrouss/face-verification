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
import cv2

#Window.clearcolor = (.87, .9, .87, 1)

Builder.load_string('''
<DisplayLayout>
    orientation: 'vertical'
    size_hint: 1, 1
''')


class LiveCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(LiveCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture

class ReferencePanel(GridLayout):
    def __init__(self, **kwargs):
        super(ReferencePanel, self).__init__(**kwargs)
        self.cols = 1

        title = Label(text='Face Verification\nApp', font_size="30sp", halign='center')
        self.add_widget(title)

        self.add_widget(Button(text='Hello'))
        self.add_widget(Button(text='Hello'))


class DisplayLayout(BoxLayout):
    def __init__(self, capture, **kwargs):
        super(DisplayLayout, self).__init__(**kwargs)
        self.capture = capture
        self.my_camera = LiveCamera(capture=capture, fps=30, size_hint=(1, 1))

        side_panel = ReferencePanel()

        self.add_widget(self.my_camera)
        self.add_widget(side_panel)
        

class MyLayout(GridLayout):
    def __init__(self, capture, **kwargs):
        super(MyLayout, self).__init__(**kwargs)
        
        self.cols = 2

        self.camera = LiveCamera(capture, fps=30)

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
        picture_select_button.bind(on_press= lambda a:print("picture chosen"))
        
        self.add_widget(self.camera)
        self.add_widget(ReferencePanel())
        self.add_widget(screenshot_button)
        self.add_widget(picture_select_button)


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
