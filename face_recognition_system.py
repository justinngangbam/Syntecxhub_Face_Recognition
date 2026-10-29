import cv2
import numpy as np
import os
import pickle

class FaceRecognitionSystem:
    def __init__(self, dataset_dir="registered_faces", embeddings_file="face_embeddings.pkl"):
        self.dataset_dir = dataset_dir
        self.embeddings_file = embeddings_file
        self.known_embeddings = {}  # Format: {"Name": [embedding1, embedding2, ...]}

        # 1. Load OpenCV's built-in Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Create dataset directory if it does not exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.load_embeddings()

    def load_embeddings(self):
        """Loads saved face embeddings from disk if available."""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "rb") as file:
                self.known_embeddings = pickle.load(file)

    def save_embeddings(self):
        """Saves registered face embeddings to disk."""
        with open(self.embeddings_file, "wb") as file:
            pickle.dump(self.known_embeddings, file)

    def extract_face_embedding(self, face_roi):
        """
        Preprocesses a cropped face image and extracts a normalized 1D feature embedding
        combining lighting-equalized spatial features and intensity histograms.
        """
        # Resize to a standard 100x100 resolution
        resized_face = cv2.resize(face_roi, (100, 100))

        # Apply CLAHE (Adaptive Histogram Equalization) to handle lighting variations
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_face = clahe.apply(resized_face)

        # Extract spatial pixel features (downsampled to 32x32) and histogram features
        spatial_features = cv2.resize(equalized_face, (32, 32)).flatten().astype(np.float32) / 255.0
        hist_features = cv2.calcHist([equalized_face], [0], None, [64], [0, 256]).flatten()
        hist_features = hist_features / (np.sum(hist_features) + 1e-7)

        # Combine into a single feature vector and apply L2 normalization
        embedding = np.concatenate([spatial_features, hist_features])
        norm = np.linalg.norm(embedding)
        return embedding / (norm + 1e-7)

    def recognize_face(self, face_roi, threshold=0.82):
        """Compares a detected face's embedding against all registered faces using Cosine Similarity."""
        if not self.known_embeddings:
            return "Unknown", 0.0

        current_embedding = self.extract_face_embedding(face_roi)
        best_name = "Unknown"
        best_score = 0.0

        for name, embeddings_list in self.known_embeddings.items():
            # Calculate cosine similarity against all stored samples for this person
            scores = [float(np.dot(current_embedding, saved_emb)) for saved_emb in embeddings_list]
            max_person_score = max(scores) if scores else 0.0

            if max_person_score > best_score:
                best_score = max_person_score
                best_name = name

        if best_score >= threshold:
            return best_name, round(best_score * 100, 1)
        return "Unknown", round(best_score * 100, 1)

    def register_new_person(self, person_name, num_samples=20):
        """Captures face samples from the webcam and registers a new person's embeddings."""
        person_folder = os.path.join(self.dataset_dir, person_name)
        os.makedirs(person_folder, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Could not access the webcam.")
            return

        print(f"\n[Info] Look at the camera. Capturing {num_samples} samples for '{person_name}'...")
        collected_embeddings = []
        sample_count = 0
        frame_counter = 0

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror view
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                # Draw bounding box around the face being registered
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

                # Capture a sample every 4 frames so there is slight natural variation
                if frame_counter % 4 == 0 and sample_count < num_samples:
                    face_roi = gray[y:y + h, x:x + w]
                    embedding = self.extract_face_embedding(face_roi)
                    collected_embeddings.append(embedding)
                    sample_count += 1

                    # Save the cropped face image to the person's folder
                    img_path = os.path.join(person_folder, f"sample_{sample_count}.jpg")
                    cv2.imwrite(img_path, face_roi)

                # Display capture progress on screen
                status_text = f"Registering {person_name}: {sample_count}/{num_samples}"
                cv2.putText(frame, status_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

            frame_counter += 1
            cv2.imshow("Register New Face - Press 'q' to Cancel", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if collected_embeddings:
            self.known_embeddings[person_name] = collected_embeddings
            self.save_embeddings()
            print(f"[Success] '{person_name}' registered with {len(collected_embeddings)} face samples!")
        else:
            print("[Warning] No face samples were captured.")

    def run_webcam_recognition(self):
        """Runs real-time multi-face detection and recognition on live webcam video."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Could not access the webcam.")
            return

        print("\n[Info] Starting live Face Detection & Recognition. Press 'q' in the video window to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect multiple faces in the current video frame
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                name, confidence = self.recognize_face(face_roi)

                # Green bounding box for known faces, Red for Unknown
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown"

                # Draw bounding box and label background
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 30), (x + w, y), color, cv2.FILLED)
                cv2.putText(frame, label, (x + 6, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display total faces detected in the top-left corner
            cv2.putText(frame, f"Faces Detected: {len(faces)} | Press 'q' to Exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Syntecxhub - Face Detection & Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_from_image(self, image_path):
        """Detects and recognizes faces in a static image file."""
        if not os.path.exists(image_path):
            print(f"[Error] Image file '{image_path}' not found.")
            return

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            name, confidence = self.recognize_face(face_roi)
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown"

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        print(f"[Info] Detected {len(faces)} face(s) in '{image_path}'. Press any key on the image window to close.")
        cv2.imshow("Image Face Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    system = FaceRecognitionSystem()

    while True:
        print("\n" + "=" * 55)
        print("   SYNTECXHUB - FACE DETECTION & RECOGNITION SYSTEM")
        print("=" * 55)
        print(" 1. Register / Add a New Person (Webcam)")
        print(" 2. Start Live Face Detection & Recognition (Webcam)")
        print(" 3. Test Face Recognition on an Image File")
        print(" 4. View Registered People")
        print(" 5. Exit")
        print("=" * 55)

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            name = input("Enter the person's name to register: ").strip()
            if name:
                system.register_new_person(name)
            else:
                print("[Warning] Name cannot be empty.")
        elif choice == "2":
            system.run_webcam_recognition()
        elif choice == "3":
            img_path = input("Enter the path to the image file: ").strip()
            system.recognize_from_image(img_path)
        elif choice == "4":
            if system.known_embeddings:
                print("\nRegistered People in Database:")
                for idx, person in enumerate(system.known_embeddings.keys(), 1):
                    print(f"  {idx}. {person} ({len(system.known_embeddings[person])} samples)")
            else:
                print("\n[Info] No people registered yet. Use Option 1 to add someone!")
        elif choice == "5":
            print("Exiting Face Recognition System. Goodbye!")
            break
        else:
            print("[Warning] Invalid choice. Please enter a number from 1 to 5.")

if __name__ == "__main__":
    main()