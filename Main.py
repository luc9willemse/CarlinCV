import Detection

if __name__ == "__main__":
    dt = Detection.Detection(r'C:\Users\User\Documents\Code\Carlin\CarlinCV\CarlinCV\videos\test_video.mp4', r'C:\Users\User\Documents\Code\Carlin\CarlinCV\CarlinCV\videos\shape.png')

    dt.process_video()