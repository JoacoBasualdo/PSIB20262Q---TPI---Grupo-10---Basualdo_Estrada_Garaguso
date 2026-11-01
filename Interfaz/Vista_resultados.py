from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class ImagenAjustable(QLabel):

    def __init__(self, imagen):
        super().__init__()
        self.original = imagen
        self.setAlignment(Qt.AlignHCenter)
        self.setPixmap(imagen)

    def ajustar_ancho(self, ancho_disponible):
        ancho = min(self.original.width(), max(ancho_disponible, 200))
        self.setPixmap(self.original.scaledToWidth(ancho, Qt.SmoothTransformation))


class PestanaEtapa(QScrollArea):

    def __init__(self, titulo, elementos):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.imagenes = []

        contenido = QWidget()
        disposicion = QVBoxLayout(contenido)

        etiqueta_titulo = QLabel(titulo)
        fuente_titulo = etiqueta_titulo.font()
        fuente_titulo.setPointSize(14)
        fuente_titulo.setBold(True)
        etiqueta_titulo.setFont(fuente_titulo)
        etiqueta_titulo.setStyleSheet("color: #02568A; padding-bottom: 6px;")
        etiqueta_titulo.setWordWrap(True)
        disposicion.addWidget(etiqueta_titulo)

        if not elementos:
            disposicion.addWidget(QLabel("Esta etapa no generó resultados con las opciones elegidas."))

        for tipo, valor in elementos:
            if tipo == "subtitulo":
                disposicion.addWidget(self._crear_subtitulo(valor))
            elif tipo == "texto":
                disposicion.addWidget(self._crear_texto(valor))
            elif tipo == "figura":
                disposicion.addWidget(self._crear_imagen(valor))

        disposicion.addStretch()
        self.setWidget(contenido)

    def _crear_subtitulo(self, texto):
        etiqueta = QLabel(texto)
        fuente = etiqueta.font()
        fuente.setPointSize(11)
        fuente.setBold(True)
        etiqueta.setFont(fuente)
        etiqueta.setStyleSheet("color: #01426B; border-bottom: 2px solid #D5DEE8; padding: 10px 0 4px 0;")
        return etiqueta

    def _crear_texto(self, texto):
        etiqueta = QLabel(texto)
        etiqueta.setTextFormat(Qt.PlainText)
        etiqueta.setFont(QFont("Consolas", 9))
        etiqueta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        etiqueta.setStyleSheet("background: #F7F9FC; border-radius: 6px; padding: 8px;")
        return etiqueta

    def _crear_imagen(self, datos_png):
        imagen = QPixmap()
        imagen.loadFromData(datos_png, "PNG")
        etiqueta = ImagenAjustable(imagen)
        self.imagenes.append(etiqueta)
        return etiqueta

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        ancho = self.viewport().width() - 30
        for imagen in self.imagenes:
            imagen.ajustar_ancho(ancho)