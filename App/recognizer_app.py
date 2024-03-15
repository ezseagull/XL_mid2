import pygame
import cv2
import pickle
from skimage.metrics import structural_similarity as ssim
import os
from faces_train import updateLBPH


# Load resource
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer.yml')

labels = {}
with open("labels.pickle", "rb") as f:
    org_labels = pickle.load(f)
    labels = {v: k for k, v in org_labels.items()}

############# APP #################
pygame.init()
pygame.display.set_caption("Face Recognize App")

cap = cv2.VideoCapture(0)
width, height = int(cap.get(3)), int(cap.get(4))
screen = pygame.display.set_mode((width, height))


# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)


class Button:

    def __init__(self, text, font, x, y, color=BLACK, bgcolor=WHITE):
        self.font = font
        self.color = color
        self.bgcolor = bgcolor
        self.surface = self.font.render(text, True, self.color, self.bgcolor)
        self.rect = self.surface.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, surface):
        surface.blit(self.surface, self.rect)


class Text:
    def __init__(self, text, font, x, y, w, h, color=BLACK):
        self.text = text
        self.surface = font.render(self.text, True, color)
        self.rect = self.surface.get_rect()
        self.rect.topleft = (x, y)
        self.rect.width = w
        self.rect.height = h
        self.color = color
        self.font = font

    def update(self, screen):
        self.surface = self.font.render(self.text, True, self.color)
        screen.blit(self.surface, self.rect)


# Define app state
states = {
    "1": "HOME_PAGE",
    "2": "RECOGNIZER",
    "3": "NEW_FACE"
}

currentState = "1"


# Home Page
def homPage():
    pygame.display.set_caption("Face Recognize App: Menu")
    global currentState
    font = pygame.font.SysFont('Arial', 30)
    option_1_button = Button('Add New Face', font, 50, 50)
    option_2_button = Button('Recognize Faces', font, 50, 150)
    running = True
    while running:
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    quit()
                case pygame.MOUSEBUTTONDOWN:
                    # Check if the user clicked on a button
                    if option_1_button.rect.collidepoint(pygame.mouse.get_pos()):
                        currentState = '3'
                        running = False
                    elif option_2_button.rect.collidepoint(pygame.mouse.get_pos()):
                        currentState = '2'
                        running = False
        screen.fill(WHITE)
        # Draw the buttons
        option_1_button.draw(screen)
        option_2_button.draw(screen)

        # Update the display
        pygame.display.flip()


# Recognizer Page
def recognizerPage():
    pygame.display.set_caption("Face Recognize App: Recognizer Faces")
    global currentState
    font = pygame.font.SysFont('Arial', 30)
    home_button = Button('<= Home', font, 5, 5)
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            rect = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(rect)

            # Draw Rect boundary
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
            f_color = (255, 255, 255)
            f_stroke = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "Unknown"
            # Recognizer
            if conf < 40:  # and conf < 85:
                # Draw Text
                # print(conf)
                name = labels[id_]
            cv2.putText(frame, name, (x, y), font, 1,
                        f_color, f_stroke, cv2.LINE_AA)

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.surfarray.make_surface(frame)

        screen.blit(frame, (0, 0))
        home_button.draw(screen)
        pygame.display.update()

        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    quit()
                case pygame.MOUSEBUTTONDOWN:
                    # Check if the user clicked on a button
                    if home_button.rect.collidepoint(pygame.mouse.get_pos()):
                        currentState = '1'
                        running = False


def saveFrames(nameFolder, colorFrames, rootFolder="../images/celeb-data/"):
    dir_path = rootFolder + nameFolder
    if os.path.exists(dir_path):
        return False
    os.makedirs(dir_path)
    index = 0
    for frame in colorFrames:
        path = dir_path + "/" + str(index) + ".png"
        cv2.imwrite(path, frame)
        index += 1
    return True


# New Face Page


def newFacePage():
    pygame.display.set_caption("Face Recognize App: Record new Face")
    global currentState
    # Saving properties
    gray_frames = []
    color_frames = []
    max_frame = 100
    save_frame = 35

    # Home button
    font = pygame.font.SysFont('Arial', 30)
    home_button = Button('<= Home', font, 5, 5)
    save_button = Button('Save', font, width-55, 5)

    # Text Field
    text = ''
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect()
    text_rect.topright = (width-360, 0)
    text_rect.width = 300
    text_rect.height = 40
    border_rect = pygame.Rect(text_rect.left - 5, text_rect.top - 5,
                              text_rect.width + 10, text_rect.height + 10)
    border_color = BLACK
    text_color = WHITE

    # Properties text
    ssimText = Text("SSIM: 0 %", font, width-130, 50, 50, 40, text_color)
    processText = Text("Process: 0 %", font, width -
                       180, 100, 50, 40, text_color)

    # Loop
    running = True
    while running:
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    quit()
                case pygame.MOUSEBUTTONDOWN:
                    # Check if the user clicked on a button
                    if home_button.rect.collidepoint(pygame.mouse.get_pos()):
                        currentState = '1'
                        running = False
                    if save_button.rect.collidepoint(pygame.mouse.get_pos()):
                        if len(gray_frames) >= max_frame and len(text) > 5:
                            saved = saveFrames(text, color_frames)
                            if saved:
                                updateLBPH(
                                    recognizer, gray_frames, text, labels)
                                currentState = '2'
                                running = False

                case pygame.KEYDOWN:
                    if event.unicode.isprintable():
                        if len(text) < 18:
                            text += event.unicode
                            text_surface = font.render(text, True, text_color)
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                        text_surface = font.render(text, True, text_color)

        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)

        if len(faces) > 0:
            face = max(faces, key=lambda x: x[2]*x[3])
            x, y, w, h = face
            rect = gray[y:y+h, x:x+w]
            if len(gray_frames) == 0:
                gray_frames.append(rect)
                color_frames.append(frame.copy())
            else:
                gray_1 = cv2.resize(gray_frames[-1], (64, 64))
                gray_2 = cv2.resize(rect, (64, 64))
                score = ssim(gray_1, gray_2)
                ssim_score = int(score * 100)
                ssimText.text = "SSIM: " + str(ssim_score) + " %"
                if ssim_score < 70 and len(gray_frames) < max_frame:
                    gray_frames.append(rect)
                    color_frames.append(frame.copy())

            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
        # Update properties
        process_per = int(len(gray_frames) / max_frame * 100)
        processText.text = "Process: " + str(process_per) + " %"

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        # Text field
        pygame.draw.rect(screen, border_color, border_rect)
        screen.blit(text_surface, text_rect)

        # SSIM field
        ssimText.update(screen)
        processText.update(screen)

        home_button.draw(screen)
        save_button.draw(screen)

        pygame.display.update()


# Main Loop
while True:
    match states[currentState]:
        case "HOME_PAGE":
            homPage()
        case "RECOGNIZER":
            recognizerPage()
        case "NEW_FACE":
            newFacePage()
