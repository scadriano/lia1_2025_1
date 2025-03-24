import cv2
import mediapipe as mp

# Inicialização da captura de vídeo
video = cv2.VideoCapture(0)

# Configuração do MediaPipe Hands
hands = mp.solutions.hands
hands_detector = hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Conversão da imagem de BGR (câmera) para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    hand_landmarks = results.multi_hand_landmarks
    height, width, _ = frame.shape

    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, hands.HAND_CONNECTIONS)

            # Extração das coordenadas dos pontos
            points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks.landmark]

            # Contagem dos dedos levantados
            finger_tips = [8, 12, 16, 20]
            count = 0

            if points[4][0] < points[2][0]:
                count += 1
            count += sum(1 for tip in finger_tips if points[tip][1] < points[tip - 2][1])

            # Desenho do retângulo e contagem na imagem
            cv2.rectangle(frame, (80, 10), (200, 110), (255, 0, 0), -1)
            cv2.putText(frame, str(count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    cv2.imshow('Imagem', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()