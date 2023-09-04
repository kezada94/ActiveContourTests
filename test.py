import cv2

def play_video_backward(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    # Start from the last frame and read the video frames backwards
    for frame_index in range(total_frames-116, -1, -1):
        # Set the frame position to the desired index
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        ret, frame = video.read()
        try:
            if frame == None:
                continue
        except:
            pass
        # Display the frame
        cv2.imshow('Video Playback', frame)

        # Wait for a specific time (e.g., 25 milliseconds)
        if cv2.waitKey(10000) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any open windows
    video.release()
    cv2.destroyAllWindows()

play_video_backward("vidd.mp4")
