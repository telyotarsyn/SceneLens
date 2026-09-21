import cv2

def main(max_index: int = 10):
    print("Searching for available OpenCV camera indices...")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            found.append(i)
            print(f"[OK] camera index {i}: {w}x{h}")
        cap.release()

    if not found:
        print("No cameras found.")
        print("If you use an iPhone on Windows, start Camo Studio first, then run this script again.")
    else:
        print(f"Available indices: {found}")
        print("Use one of them with: python src/live_detect.py --camera INDEX")

if __name__ == "__main__":
    main()
