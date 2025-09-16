from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.snackbar import MDSnackbar
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.graphics.texture import Texture
import cv2, numpy as np, os
from ultralytics import YOLO

MODEL_PATH = os.path.join("resultados", "modelo_treinado.pt")

class ItemRow(MDBoxLayout):
    """
    Representa uma linha com o nome do item detectado e um campo para alterar a quantidade manualmente.
    """
    def __init__(self, name, quantity, **kwargs):
        """
        Inicializa a linha do item com nome e quantidade.

        Parâmetros
        ----------
        name : str
            Nome do item detectado.
        quantity : int
            Quantidade detectada inicialmente do item.
        """
        super().__init__(orientation='horizontal', size_hint_y=None, height=50, spacing=10, **kwargs)
        self.name = name
        self.label = MDLabel(text=name, halign="left", size_hint_x=0.6)
        self.input = MDTextField(text=str(quantity), input_filter='int', size_hint_x=0.4)
        self.add_widget(self.label)
        self.add_widget(self.input)

    def get_quantity(self):
        """
        Retorna a quantidade inserida no campo de texto.

        Retorno
        -------
        int
            Quantidade do item (0 se não for um número válido).
        """
        try:
            return int(self.input.text)
        except ValueError:
            return 0

class MainApp(MDApp):
    """
    Aplicativo principal que utiliza um modelo YOLO para detectar objetos e exibir os resultados em uma interface KivyMD.
    """

    def build(self):
        """
        Constrói a interface gráfica do aplicativo.

        Retorno
        -------
        MDBoxLayout
            Layout raiz da aplicação.
        """
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        self.model = YOLO(MODEL_PATH)
        self.detected_items = {}
        self.capture = None

        self.root = MDBoxLayout(orientation='vertical', padding=[20, 30, 20, 20], spacing=15)

        # Campo de IP + botão de conexão
        ip_layout = MDBoxLayout(size_hint_y=None, height=50, spacing=10)
        self.url_input = MDTextField(hint_text="Camera IP URL (leave empty for default)")
        self.btn_connect = MDRaisedButton(text="Connect", icon="access-point", on_release=self.connect_camera)
        ip_layout.add_widget(self.url_input)
        ip_layout.add_widget(self.btn_connect)
        self.root.add_widget(ip_layout)

        # Área de vídeo
        self.img = Image(size_hint_y=0.6)
        self.root.add_widget(self.img)

        # Botões de ação
        from kivy.uix.widget import Widget
        btn_layout = MDBoxLayout(orientation='horizontal', size_hint_y=None, height=60, padding=(0, 10), spacing=20)
        left_spacer = Widget(size_hint_x=1)
        right_spacer = Widget(size_hint_x=1)

        self.btn_capture = MDRaisedButton(text="Capture", icon="camera", on_release=self.capture_frame, size_hint_x=None, width=130)
        self.btn_clear = MDRaisedButton(text="Clear", icon="refresh", on_release=self.clear_list, size_hint_x=None, width=130)
        self.btn_copy = MDRaisedButton(text="Copy List", icon="content-copy", on_release=self.copy_list, size_hint_x=None, width=130)

        btn_layout.add_widget(left_spacer)
        btn_layout.add_widget(self.btn_capture)
        btn_layout.add_widget(self.btn_clear)
        btn_layout.add_widget(self.btn_copy)
        btn_layout.add_widget(right_spacer)

        self.root.add_widget(btn_layout)

        # Lista de itens
        self.scroll = MDScrollView(size_hint=(1, 0.4))
        self.grid = MDGridLayout(cols=1, size_hint_y=None, spacing=10, padding=10)
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.scroll.add_widget(self.grid)
        self.root.add_widget(self.scroll)

        return self.root

    def show_snackbar(self, message):
        """
        Exibe uma mensagem temporária na tela.

        Parâmetros
        ----------
        message : str
            Texto a ser exibido no snackbar.
        """
        snackbar = MDSnackbar(
            MDLabel(text=message, halign="center"),
            y=24,
            pos_hint={"center_x": 0.5},
            size_hint_x=0.8,
        )
        snackbar.open()

    def connect_camera(self, instance):
        """
        Conecta à câmera IP ou webcam local e inicia o streaming.

        Parâmetros
        ----------
        instance : Widget
            Referência ao botão que acionou a função.
        """
        url = self.url_input.text.strip()
        self.capture = cv2.VideoCapture(url if url else 0)
        if not self.capture.isOpened():
            self.show_snackbar("Failed to connect to camera!")
            return
        Clock.schedule_interval(self.update_stream, 1.0 / 60.0)
        self.show_snackbar("Camera connected successfully!")

    def update_stream(self, dt):
        """
        Atualiza o frame da câmera e aplica detecção do modelo YOLO.

        Parâmetros
        ----------
        dt : float
            Tempo decorrido desde a última chamada (gerenciado pelo Kivy Clock).
        """
        if self.capture:
            ret, frame = self.capture.read()
            if not ret:
                return

            results = self.model(frame)

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0]) * 100
                    if conf < 50:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    name = result.names[cls_id]
                    label = f"{name} ({conf:.1f}%)"

                    # Caixa de detecção
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Texto sobreposto com fundo
                    font_scale, font_thickness = 2.1, 6
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 0, 0), -1)
                    cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                    alpha = 0.5
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Atualiza o widget de imagem no app
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture

    def capture_frame(self, instance):
        """
        Captura um frame atual da câmera e atualiza os itens detectados.

        Parâmetros
        ----------
        instance : Widget
            Referência ao botão que acionou a função.
        """
        if not self.capture:
            self.show_snackbar("No camera connected!")
            return

        ret, frame = self.capture.read()
        if not ret:
            self.show_snackbar("Failed to capture image!")
            return

        results = self.model(frame)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0]) * 100
                if conf < 50:
                    continue
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                self.detected_items[name] = self.detected_items.get(name, 0) + 1

        self.update_list()

    def update_list(self):
        """
        Atualiza a lista de itens detectados na interface.
        """
        self.grid.clear_widgets()
        for name, quantity in self.detected_items.items():
            item_row = ItemRow(name, quantity)
            self.grid.add_widget(item_row)

    def copy_list(self, instance):
        """
        Copia a lista de itens com quantidades para a área de transferência.

        Parâmetros
        ----------
        instance : Widget
            Referência ao botão que acionou a função.
        """
        text = ""
        for child in self.grid.children:
            if isinstance(child, ItemRow):
                qty = child.get_quantity()
                if qty > 0:
                    text += f"{child.name}: {qty}\n"
        if text:
            Clipboard.copy(text)
            self.show_snackbar("List copied successfully!")
        else:
            self.show_snackbar("No valid items to copy.")

    def clear_list(self, instance):
        """
        Limpa a lista de itens detectados.

        Parâmetros
        ----------
        instance : Widget
            Referência ao botão que acionou a função.
        """
        self.detected_items.clear()
        self.grid.clear_widgets()
        self.show_snackbar("List cleared!")

if __name__ == "__main__":
    MainApp().run()
